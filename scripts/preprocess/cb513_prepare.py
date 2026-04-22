#!/usr/bin/env python3
"""Prepare CB513 canonical dataset artifacts.

This script parses raw CB513 `.all` files, maps DSSP labels from Q8-like symbols
to Q3 labels, validates integrity, and writes canonical artifacts for downstream
training/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple


AA20 = "ACDEFGHIKLMNPQRSTVWY"
DSSP_KNOWN_SYMBOLS = set("HBEGITS_")
Q3_SYMBOLS = ("H", "E", "C")
Q3_TO_ID = {"H": 0, "E": 1, "C": 2}

# Standard BLOSUM62 log-odds matrix (row order = AA20 alphabet).
# Each row gives substitution scores for that amino acid against all 20 AAs.
# Source: NCBI BLOSUM62, stored as integers (common convention).
BLOSUM62: Dict[str, List[int]] = {
    #        A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    "A": [   4, -1, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],
    "C": [  -1,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    "D": [  -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],
    "E": [  -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],
    "F": [  -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],
    "G": [   0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],
    "H": [  -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],
    "I": [  -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],
    "K": [  -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],
    "L": [  -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],
    "M": [  -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],
    "N": [  -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],
    "P": [  -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],
    "Q": [  -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],
    "R": [  -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],
    "S": [   1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],
    "T": [   0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],
    "V": [   0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],
    "W": [  -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],
    "Y": [  -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],
}


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    dssp_q8: str
    labels_q3: str
    length: int
    source_file: str


def map_dssp_to_q3(symbol: str) -> str:
    """Map one DSSP symbol to Q3 according to paper rule."""
    if symbol in {"H", "G", "I"}:
        return "H"
    if symbol in {"B", "E"}:
        return "E"
    return "C"


def map_q3_to_id(symbol: str) -> int:
    """Map Q3 label to class id."""
    if symbol not in Q3_TO_ID:
        raise ValueError(f"Invalid Q3 label '{symbol}'. Expected one of {Q3_SYMBOLS}.")
    return Q3_TO_ID[symbol]


def validate_lengths(record: ProteinRecord) -> None:
    """Validate sequence/label length consistency for one record."""
    if len(record.sequence) != len(record.dssp_q8):
        raise ValueError(
            f"Length mismatch in {record.source_file}: "
            f"RES={len(record.sequence)} DSSP={len(record.dssp_q8)}"
        )
    if len(record.labels_q3) != len(record.sequence):
        raise ValueError(
            f"Length mismatch in {record.source_file}: "
            f"Q3={len(record.labels_q3)} RES={len(record.sequence)}"
        )


def parse_bool(value: str) -> bool:
    value_norm = value.strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'.")


def normalize_symbol_line(raw: str) -> str:
    """Normalize symbol list line by removing commas and whitespace."""
    return "".join(ch for ch in raw if ch not in {",", " ", "\t", "\r", "\n"})


def parse_cb513_file(path: Path, raw_root: Path) -> Tuple[ProteinRecord, Counter, Counter, int, List[str]]:
    """Parse one `.all` file into canonical record and counters.

    Returns:
      record, q8_counter, q3_counter, unknown_to_c_count, warnings
    """
    sequence_line = None
    dssp_line = None

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("RES:"):
                sequence_line = line.split(":", 1)[1]
            elif line.startswith("DSSP:"):
                dssp_line = line.split(":", 1)[1]

            if sequence_line is not None and dssp_line is not None:
                break

    if sequence_line is None:
        raise ValueError(f"Missing RES line in {path}")
    if dssp_line is None:
        raise ValueError(f"Missing DSSP line in {path}")

    sequence = normalize_symbol_line(sequence_line)
    dssp_q8 = normalize_symbol_line(dssp_line)

    protein_id = path.name.rsplit(".all", 1)[0]
    labels_q3_chars: List[str] = []
    q8_counts = Counter()
    q3_counts = Counter()
    unknown_to_c_count = 0
    unknown_symbols = Counter()

    for char in dssp_q8:
        q8_counts[char] += 1
        mapped = map_dssp_to_q3(char)
        labels_q3_chars.append(mapped)
        q3_counts[mapped] += 1
        if char not in DSSP_KNOWN_SYMBOLS:
            unknown_to_c_count += 1
            unknown_symbols[char] += 1

    labels_q3 = "".join(labels_q3_chars)
    source_file = str(path.relative_to(raw_root))
    record = ProteinRecord(
        protein_id=protein_id,
        sequence=sequence,
        dssp_q8=dssp_q8,
        labels_q3=labels_q3,
        length=len(sequence),
        source_file=source_file,
    )
    validate_lengths(record)

    warnings = []
    if unknown_symbols:
        unknown_detail = ", ".join(f"{k}:{v}" for k, v in sorted(unknown_symbols.items()))
        warnings.append(f"{source_file}: unknown DSSP symbols mapped to C -> {unknown_detail}")

    return record, q8_counts, q3_counts, unknown_to_c_count, warnings


def make_window(sequence: str, index: int, window_size: int = 13, padding_token: str = "X") -> str:
    """Build centered fixed-size window for one residue index."""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be positive odd integer, got {window_size}")
    if not (0 <= index < len(sequence)):
        raise IndexError(f"index {index} out of bounds for sequence length {len(sequence)}")

    radius = window_size // 2
    chars: List[str] = []
    for pos in range(index - radius, index + radius + 1):
        if 0 <= pos < len(sequence):
            chars.append(sequence[pos])
        else:
            chars.append(padding_token)
    return "".join(chars)


def encode_one_hot_window(window: str) -> List[float]:
    """Encode one window into flattened 20 x len(window) one-hot vector.

    Ambiguous amino acids (`B/X/Z`) and unknown tokens map to all-zero vectors.
    """
    vec: List[float] = []
    for aa in window:
        col = [0.0] * len(AA20)
        aa_idx = AA20.find(aa)
        if aa_idx >= 0:
            col[aa_idx] = 1.0
        vec.extend(col)
    return vec


def encode_blosum62_window(window: str) -> List[float]:
    """Encode one window using BLOSUM62 substitution scores.

    Each residue maps to its 20-dim BLOSUM62 row (float cast).
    Ambiguous amino acids (B/X/Z) and unknown tokens map to all-zero vectors,
    matching the one-hot policy for consistency.
    """
    vec: List[float] = []
    for aa in window:
        if aa in BLOSUM62:
            vec.extend(float(v) for v in BLOSUM62[aa])
        else:
            vec.extend([0.0] * len(AA20))
    return vec


def write_binary_split(out_dir: Path, split_name: str, X_rows: List[List[float]], y_rows: List[int]) -> None:
    """Write flat binary files for one split.

    X file: row-major float32, shape [N, feature_dim].
    y file: contiguous int32, shape [N].
    No in-file headers — shapes are recorded in binary_meta.json.

    C loading pattern:
        fread(X, sizeof(float), N * feature_dim, fx);
        fread(y, sizeof(int),   N,               fy);
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"{split_name}_X.bin"
    y_path = out_dir / f"{split_name}_y.bin"

    with x_path.open("wb") as fx:
        for row in X_rows:
            fx.write(struct.pack(f"{len(row)}f", *row))

    with y_path.open("wb") as fy:
        for label in y_rows:
            fy.write(struct.pack("i", label))


def generate_windows_for_record(
    record: ProteinRecord,
    window_size: int = 13,
    padding_token: str = "X",
    encoder: Callable[[str], List[float]] = encode_one_hot_window,
) -> Iterator[Tuple[List[float], int]]:
    """Yield (encoded_feature_vector, class_id) for each residue."""
    for idx, label in enumerate(record.labels_q3):
        window = make_window(record.sequence, idx, window_size=window_size, padding_token=padding_token)
        yield encoder(window), map_q3_to_id(label)


def split_fixed(
    protein_ids: Sequence[str], seed: int, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Dict[str, List[str]]:
    """Deterministic protein-level fixed split."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0,1)")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    shuffled = list(protein_ids)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)
    n_test = n_total - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total
    assert len(test_ids) == n_test

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def ensure_q3_alphabet(records: Iterable[ProteinRecord]) -> None:
    """Ensure mapped labels only contain H/E/C."""
    allowed = set(Q3_SYMBOLS)
    for rec in records:
        bad = sorted(set(rec.labels_q3) - allowed)
        if bad:
            raise ValueError(f"Invalid Q3 symbols in {rec.source_file}: {bad}")


def write_tsv(path: Path, records: Sequence[ProteinRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["protein_id", "sequence", "dssp_q8", "labels_q3", "length", "source_file"]
        )
        for rec in records:
            writer.writerow(
                [rec.protein_id, rec.sequence, rec.dssp_q8, rec.labels_q3, rec.length, rec.source_file]
            )


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CB513 canonical dataset artifacts.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/cb513/513_distribute"),
        help="Path to raw CB513 .all files directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/cb513"),
        help="Output root directory for processed artifacts.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=13,
        help="Window size for compatibility checks and downstream helpers (must be odd).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="fixed",
        choices=["fixed"],
        help="Dataset split protocol. Currently only 'fixed' is implemented.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split generation.",
    )
    parser.add_argument(
        "--keep-tsv",
        type=parse_bool,
        default=True,
        help="Whether to persist canonical TSV (true/false). Default: true.",
    )
    parser.add_argument(
        "--write-binary",
        type=parse_bool,
        default=True,
        help="Write flat binary split files for C training (true/false). Default: true.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="onehot",
        choices=["onehot", "blosum62"],
        help="Feature encoding scheme: 'onehot' (default) or 'blosum62'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.window_size <= 0 or args.window_size % 2 == 0:
        raise ValueError(f"--window-size must be a positive odd integer, got {args.window_size}")

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw path is not a directory: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.all"))
    if not raw_files:
        raise FileNotFoundError(f"No .all files found under {raw_dir}")

    # Select encoder based on CLI flag.
    encoder: Callable[[str], List[float]]
    if args.encoding == "blosum62":
        encoder = encode_blosum62_window
        feature_dim = len(AA20) * args.window_size
    else:
        encoder = encode_one_hot_window
        feature_dim = len(AA20) * args.window_size

    records: List[ProteinRecord] = []
    global_q8_counts: Counter = Counter()
    global_q3_counts: Counter = Counter()
    unknown_to_c_count = 0
    warnings: List[str] = []

    for path in raw_files:
        record, q8_counts, q3_counts, unknown_count, file_warnings = parse_cb513_file(path, raw_dir)
        records.append(record)
        global_q8_counts.update(q8_counts)
        global_q3_counts.update(q3_counts)
        unknown_to_c_count += unknown_count
        warnings.extend(file_warnings)

    ensure_q3_alphabet(records)

    total_residues = sum(rec.length for rec in records)
    protein_ids = [rec.protein_id for rec in records]
    record_by_id = {rec.protein_id: rec for rec in records}

    if args.split_mode != "fixed":
        raise ValueError(f"Unsupported split mode: {args.split_mode}")
    splits = split_fixed(protein_ids, seed=args.seed)

    split_payload = {
        "split_mode": "fixed",
        "seed": args.seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "counts": {k: len(v) for k, v in splits.items()},
        "ids": splits,
    }

    # Generate encoded windows per split and compute per-split class distribution.
    ID_TO_Q3 = {v: k for k, v in Q3_TO_ID.items()}
    split_X: Dict[str, List[List[float]]] = {}
    split_y: Dict[str, List[int]] = {}
    split_class_dist: Dict[str, Dict[str, int]] = {}
    window_sample_count = 0

    for split_name, pid_list in splits.items():
        X_rows: List[List[float]] = []
        y_rows: List[int] = []
        class_counts: Counter = Counter()
        for pid in pid_list:
            rec = record_by_id[pid]
            for feat_vec, class_id in generate_windows_for_record(
                rec, window_size=args.window_size, padding_token="X", encoder=encoder
            ):
                X_rows.append(feat_vec)
                y_rows.append(class_id)
                class_counts[ID_TO_Q3[class_id]] += 1
                window_sample_count += 1
        split_X[split_name] = X_rows
        split_y[split_name] = y_rows
        split_class_dist[split_name] = {q: class_counts[q] for q in Q3_SYMBOLS}

    if window_sample_count != total_residues:
        raise RuntimeError(
            f"Window sample count mismatch: got {window_sample_count}, expected {total_residues}"
        )

    # Print per-split class distribution table.
    print("\nPer-split class distribution:")
    header = f"  {'Split':<8} | {'H':>8} | {'E':>8} | {'C':>8} | {'Total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sname in ("train", "val", "test"):
        dist = split_class_dist[sname]
        total = sum(dist.values())
        print(f"  {sname:<8} | {dist['H']:>8,} | {dist['E']:>8,} | {dist['C']:>8,} | {total:>8,}")
    global_h = sum(split_class_dist[s]["H"] for s in splits)
    global_e = sum(split_class_dist[s]["E"] for s in splits)
    global_c = sum(split_class_dist[s]["C"] for s in splits)
    print(f"  {'TOTAL':<8} | {global_h:>8,} | {global_e:>8,} | {global_c:>8,} | {total_residues:>8,}")
    print()

    metadata_payload = {
        "dataset": "CB513",
        "raw_dir": str(raw_dir),
        "source_files_count": len(raw_files),
        "protein_count": len(records),
        "total_residues": total_residues,
        "class_mapping": Q3_TO_ID,
        "q8_symbol_counts": dict(sorted(global_q8_counts.items())),
        "q3_symbol_counts": dict(sorted(global_q3_counts.items())),
        "unknown_dssp_to_c_count": unknown_to_c_count,
        "files_with_warnings": warnings,
        "files_with_errors": [],
        "split_mode": args.split_mode,
        "split_seed": args.seed,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_class_distribution": split_class_dist,
        "window_size": args.window_size,
        "window_sample_count_check": window_sample_count,
        "encoding": args.encoding,
        "feature_dim": feature_dim,
        "one_hot_feature_dim": len(AA20) * args.window_size,
        "aa20_alphabet": AA20,
        "ambiguous_aa_policy": "B/X/Z and unknown sequence tokens map to all-zero 20-d vector",
        "padding_policy": "centered window with X padding; X maps to all-zero 20-d vector",
    }

    canonical_dir = out_dir / "canonical"
    split_dir = out_dir / "splits"
    binary_dir = out_dir / "binary"

    if args.keep_tsv:
        write_tsv(canonical_dir / "proteins.tsv", records)
    write_json(canonical_dir / "metadata.json", metadata_payload)
    write_json(split_dir / "fixed_split.json", split_payload)

    if args.write_binary:
        binary_meta: Dict = {
            "feature_dim": feature_dim,
            "num_classes": len(Q3_SYMBOLS),
            "class_map": Q3_TO_ID,
            "dtype_X": "float32",
            "dtype_y": "int32",
            "encoding": args.encoding,
            "splits": {},
        }
        for sname in ("train", "val", "test"):
            write_binary_split(binary_dir, sname, split_X[sname], split_y[sname])
            n = len(split_y[sname])
            binary_meta["splits"][sname] = {
                "n_samples": n,
                "X_shape": [n, feature_dim],
                "class_distribution": split_class_dist[sname],
            }
        write_json(binary_dir / "binary_meta.json", binary_meta)
        print(f"  Binary files written: {binary_dir}")

    print("CB513 preprocessing complete.")
    print(f"  Parsed proteins     : {len(records)}")
    print(f"  Total residues      : {total_residues}")
    print(f"  Unknown DSSP -> C   : {unknown_to_c_count}")
    print(f"  Window sample count : {window_sample_count}")
    print(f"  Encoding            : {args.encoding}")
    print(f"  Feature dim         : {feature_dim}")
    print(f"  Output root         : {out_dir}")


if __name__ == "__main__":
    main()
