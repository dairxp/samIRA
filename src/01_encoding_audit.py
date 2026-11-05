import os
import sys

# Ensure UTF-8 output in Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")

with open(raw_path, "rb") as f:
    raw_bytes = f.read()

print(f"Total byte length: {len(raw_bytes):,} bytes")
print(f"Starts with UTF-8 BOM: {raw_bytes.startswith(b'\xef\xbb\xbf')}")

# Find all non-ascii byte occurrences
non_ascii_indices = [i for i, b in enumerate(raw_bytes) if b >= 128]
print(f"Total non-ASCII bytes: {len(non_ascii_indices):,}")

# Check what sequences of non-ASCII bytes exist
i = 0
sequences = {}
while i < len(raw_bytes):
    if raw_bytes[i] >= 128:
        start = i
        while i < len(raw_bytes) and raw_bytes[i] >= 128:
            i += 1
        seq = raw_bytes[start:i]
        sequences[seq] = sequences.get(seq, 0) + 1
    else:
        i += 1

print("\nUnique non-ASCII byte sequences found:")
for seq, count in sorted(sequences.items(), key=lambda x: x[1], reverse=True):
    print(f"  Sequence: {repr(seq)} | Hex: {seq.hex(' ')} | Count: {count:,}")

# Look at lines containing \xef\xbf\xbd
lines = raw_bytes.splitlines()
print(f"\nTotal lines: {len(lines)}")
print("\nFirst 10 lines with non-ASCII sequences:")
sample_count = 0
for idx, line in enumerate(lines):
    if any(b >= 128 for b in line):
        # exclude line 0 if it only has BOM
        if idx == 0 and line.startswith(b'\xef\xbb\xbf') and not any(b >= 128 for b in line[3:]):
            continue
        sample_count += 1
        print(f"Line {idx+1}: {line.decode('utf-8', errors='replace')}")
        if sample_count >= 10:
            break
