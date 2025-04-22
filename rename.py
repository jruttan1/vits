# rename_and_update.py
import os, re, unicodedata

wav_dir = "data/wavs"
lists   = ["data/train.txt.cleaned", "data/val.txt.cleaned"]

# 1) Build a mapping of old→new names and rename the files
mapping = {}
for fname in os.listdir(wav_dir):
    # Normalize to ASCII, drop combining marks
    nfkd = unicodedata.normalize("NFKD", fname)
    ascii_only = "".join(c for c in nfkd if ord(c) < 128)
    # Replace any character that is NOT alnum, dash, dot, underscore, space, or square bracket
    newname = re.sub(r"[^A-Za-z0-9\-\._\[\] ]+", "_", ascii_only)
    mapping[fname] = newname
    if newname != fname:
        os.rename(os.path.join(wav_dir, fname), os.path.join(wav_dir, newname))

# 2) Rewrite your filelists to point at the new names
for list_path in lists:
    out_lines = []
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            path, text = line.rstrip("\n").split("|", 1)
            fname = os.path.basename(path)
            new_fname = mapping.get(fname, fname)
            new_path = os.path.join(wav_dir, new_fname)
            out_lines.append(f"{new_path}|{text}")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

print("Done renaming WAVs and updating filelists.")