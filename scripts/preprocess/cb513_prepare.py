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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


AA20 = "ACDEFGHIKLMNPQRSTVWY"
DSSP_KNOWN_SYMBOLS = set("HBEGITS_")
Q3_SYMBOLS = ("H", "E", "C")
Q3_TO_ID = {"H": 0, "E": 1, "C": 2}


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


def generate_windows_for_record(
    record: ProteinRecord, window_size: int = 13, padding_token: str = "X"
) -> Iterator[Tuple[str, str]]:
    """Yield (window, q3_label) for each residue."""
    for idx, label in enumerate(record.labels_q3):
        yield make_window(record.sequence, idx, window_size=window_size, padding_token=padding_token), label


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

    window_sample_count = 0
    for rec in records:
        for _window, label in generate_windows_for_record(rec, window_size=args.window_size, padding_token="X"):
            _ = map_q3_to_id(label)
            window_sample_count += 1

    if window_sample_count != total_residues:
        raise RuntimeError(
            f"Window sample count mismatch: got {window_sample_count}, expected {total_residues}"
        )

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
        "window_size": args.window_size,
        "window_sample_count_check": window_sample_count,
        "one_hot_feature_dim": len(AA20) * args.window_size,
        "aa20_alphabet": AA20,
        "ambiguous_aa_policy": "B/X/Z and unknown sequence tokens map to all-zero 20-d vector",
        "padding_policy": "centered window with X padding; X maps to all-zero 20-d vector",
    }

    canonical_dir = out_dir / "canonical"
    split_dir = out_dir / "splits"

    if args.keep_tsv:
        write_tsv(canonical_dir / "proteins.tsv", records)
    write_json(canonical_dir / "metadata.json", metadata_payload)
    write_json(split_dir / "fixed_split.json", split_payload)

    print("CB513 preprocessing complete.")
    print(f"  Parsed proteins     : {len(records)}")
    print(f"  Total residues      : {total_residues}")
    print(f"  Unknown DSSP -> C   : {unknown_to_c_count}")
    print(f"  Window sample count : {window_sample_count}")
    print(f"  Output root         : {out_dir}")


if __name__ == "__main__":
    main()
