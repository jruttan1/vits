import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import commons
import unicodedata
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class TextAudioLoader(Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    Applies a filter to drop examples too short for a training segment.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        # Text constraints
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # segment constraints
        self.segment_size = getattr(hparams, "segment_size", None)
        self.min_frames = self.segment_size // self.hop_length if self.segment_size else None

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        filtered = []
        lengths = []
        for wav_path, text in self.audiopaths_and_text:
            # text length check
            if not (self.min_text_len <= len(text) <= self.max_text_len):
                continue
            # approximate lengths
            file_size = os.path.getsize(wav_path)
            audio_len = file_size // 2
            mel_len = audio_len // self.hop_length
            # waveform and mel-frame segment check
            if self.segment_size and audio_len < self.segment_size:
                continue
            if self.min_frames and mel_len < self.min_frames:
                continue
            filtered.append([wav_path, text])
            lengths.append(mel_len)
        self.audiopaths_and_text = filtered
        self.lengths = lengths

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, idx):
        wav_path, text = self.audiopaths_and_text[idx]
        text_seq = self.get_text(text)
        spec, wav = self.get_audio(wav_path)
        return text_seq, spec, wav

    def get_audio(self, filename):
        audio, sr = load_wav_to_torch(filename)
        if sr != self.sampling_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio / self.max_wav_value
        # collapse multi-channel -> mono
        if audio_norm.dim() == 2:
            audio_norm = audio_norm.mean(dim=0)
        # ensure shape [1, time]
        audio_norm = audio_norm.unsqueeze(0)

        spec_path = filename.replace('.wav', '.spec.pt')
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False
            )
            spec = spec.squeeze(0)
            torch.save(spec, spec_path)
        return spec, audio_norm

    def get_text(self, text):
        # normalize Unicode: decompose accents and remove combining marks
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        if self.cleaned_text:
            seq = cleaned_text_to_sequence(text)
        else:
            seq = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            seq = commons.intersperse(seq, 0)
        return torch.LongTensor(seq)


class TextAudioCollate:
    """ Zero-pad text, mel-spec, and waveform to the max length in batch. """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, order = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            descending=True
        )
        max_text = max([x[0].size(0) for x in batch])
        max_spec = max([x[1].size(1) for x in batch])
        max_wav = max([x[2].size(1) for x in batch])

        text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec)
        wav_padded = torch.zeros(len(batch), 1, max_wav)
        text_lens = torch.zeros(len(batch), dtype=torch.long)
        spec_lens = torch.zeros(len(batch), dtype=torch.long)
        wav_lens = torch.zeros(len(batch), dtype=torch.long)

        for i, idx in enumerate(order):
            t, m, w = batch[idx]
            text_padded[i, :t.size(0)] = t
            spec_padded[i, :, :m.size(1)] = m
            wav_padded[i, :, :w.size(1)] = w
            text_lens[i] = t.size(0)
            spec_lens[i] = m.size(1)
            wav_lens[i] = w.size(1)

        if self.return_ids:
            return text_padded, text_lens, spec_padded, spec_lens, wav_padded, wav_lens, order
        return text_padded, text_lens, spec_padded, spec_lens, wav_padded, wav_lens


class TextAudioSpeakerLoader(Dataset):
    """
    Multi-speaker loader: loads audio, speaker_id, text.
    Filter too-short like TextAudioLoader.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        self.segment_size = getattr(hparams, "segment_size", None)
        self.min_frames = self.segment_size // self.hop_length if self.segment_size else None

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        filtered = []
        lengths = []
        for wav_path, sid, text in self.audiopaths_and_text:
            if not (self.min_text_len <= len(text) <= self.max_text_len):
                continue
            file_size = os.path.getsize(wav_path)
            audio_len = file_size // 2
            mel_len = audio_len // self.hop_length
            if self.segment_size and audio_len < self.segment_size:
                continue
            if self.min_frames and mel_len < self.min_frames:
                continue
            filtered.append([wav_path, sid, text])
            lengths.append(mel_len)
        self.audiopaths_and_text = filtered
        self.lengths = lengths

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, idx):
        wav_path, sid, text = self.audiopaths_and_text[idx]
        text_seq = self.get_text(text)
        spec, wav = self.get_audio(wav_path)
        sid_tensor = torch.LongTensor([int(sid)])
        return text_seq, spec, wav, sid_tensor

    def get_audio(self, filename):
        audio, sr = load_wav_to_torch(filename)
        if sr != self.sampling_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio / self.max_wav_value
        if audio_norm.dim() == 2:
            audio_norm = audio_norm.mean(dim=0)
        audio_norm = audio_norm.unsqueeze(0)
        spec_path = filename.replace('.wav', '.spec.pt')
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False
            )
            spec = spec.squeeze(0)
            torch.save(spec, spec_path)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            seq = cleaned_text_to_sequence(text)
        else:
            seq = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            seq = commons.intersperse(seq, 0)
        return torch.LongTensor(seq)


class TextAudioSpeakerCollate:
    """ Zero-pad text, mel-spec, wav, and speaker ID """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, order = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            descending=True
        )
        max_text = max([x[0].size(0) for x in batch])
        max_spec = max([x[1].size(1) for x in batch])
        max_wav = max([x[2].size(1) for x in batch])

        text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec)
        wav_padded = torch.zeros(len(batch), 1, max_wav)
        sid_padded = torch.zeros(len(batch), dtype=torch.long)

        text_lens = torch.zeros(len(batch), dtype=torch.long)
        spec_lens = torch.zeros(len(batch), dtype=torch.long)
        wav_lens = torch.zeros(len(batch), dtype=torch.long)

        for i, idx in enumerate(order):
            t, m, w, s = batch[idx]
            text_padded[i, :t.size(0)] = t
            spec_padded[i, :, :m.size(1)] = m
            wav_padded[i, :, :w.size(1)] = w
            sid_padded[i] = s
            text_lens[i] = t.size(0)
            spec_lens[i] = m.size(1)
            wav_lens[i] = w.size(1)

        if self.return_ids:
            return text_padded, text_lens, spec_padded, spec_lens, wav_padded, wav_lens, sid_padded, order
        return text_padded, text_lens, spec_padded, spec_lens, wav_padded, wav_lens, sid_padded


class DistributedBucketSampler(DistributedSampler):
    """
    Bucket sampler that groups examples by similar length and ensures each batch is filled.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for idx, length in enumerate(self.lengths):
            for b in range(len(self.boundaries) - 1):
                if self.boundaries[b] < length <= self.boundaries[b+1]:
                    buckets[b].append(idx)
                    break
        for b in reversed(range(len(buckets))):
            if not buckets[b]:
                buckets.pop(b)
                self.boundaries.pop(b+1)
        num_samples = []
        total_batch = self.num_replicas * self.batch_size
        for bucket in buckets:
            rem = (total_batch - (len(bucket) % total_batch)) % total_batch
            num_samples.append(len(bucket) + rem)
        return buckets, num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        all_batches = []
        for bucket, num_samples in zip(self.buckets, self.num_samples_per_bucket):
            if self.shuffle:
                indices = torch.randperm(len(bucket), generator=g).tolist()
            else:
                indices = list(range(len(bucket)))
            repeat, extra = divmod(num_samples, len(bucket))
            indices = indices * repeat + indices[:extra]
            indices = indices[self.rank::self.num_replicas]
            for i in range(0, len(indices), self.batch_size):
                batch = [bucket[j] for j in indices[i:i+self.batch_size]]
                all_batches.append(batch)
        if self.shuffle:
            all_batches = [all_batches[i] for i in torch.randperm(len(all_batches), generator=g)]
        assert len(all_batches) * self.batch_size == self.num_samples
        return iter(all_batches)

    def __len__(self):
        return self.num_samples // self.batch_size
