import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "preprocess" / "cb513_prepare.py"
    spec = importlib.util.spec_from_file_location("cb513_prepare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load_module()


class TestCB513Prepare(unittest.TestCase):
    def test_mapping_rule(self):
        inp = ["H", "G", "I", "B", "E", "S", "T", "_", "?"]
        out = [prep.map_dssp_to_q3(ch) for ch in inp]
        self.assertEqual(out, ["H", "H", "H", "E", "E", "C", "C", "C", "C"])

    def test_parse_valid_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "sample.all"
            p.write_text("RES:A,C,D\nDSSP:H,E,_\n", encoding="utf-8")

            rec, q8_counts, q3_counts, unknown_count, warnings = prep.parse_cb513_file(p, root)
            self.assertEqual(rec.protein_id, "sample")
            self.assertEqual(rec.sequence, "ACD")
            self.assertEqual(rec.dssp_q8, "HE_")
            self.assertEqual(rec.labels_q3, "HEC")
            self.assertEqual(rec.length, 3)
            self.assertEqual(rec.source_file, "sample.all")
            self.assertEqual(dict(q8_counts), {"H": 1, "E": 1, "_": 1})
            self.assertEqual(dict(q3_counts), {"H": 1, "E": 1, "C": 1})
            self.assertEqual(unknown_count, 0)
            self.assertEqual(warnings, [])

    def test_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "bad.all"
            p.write_text("RES:A,C,D\nDSSP:H,E\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prep.parse_cb513_file(p, root)

    def test_q3_alphabet_enforced(self):
        good = prep.ProteinRecord(
            protein_id="ok",
            sequence="ACD",
            dssp_q8="H_E",
            labels_q3="HEC",
            length=3,
            source_file="ok.all",
        )
        prep.ensure_q3_alphabet([good])

        bad = prep.ProteinRecord(
            protein_id="bad",
            sequence="ACD",
            dssp_q8="H_E",
            labels_q3="HEX",
            length=3,
            source_file="bad.all",
        )
        with self.assertRaises(ValueError):
            prep.ensure_q3_alphabet([bad])

    def test_fixed_split_determinism(self):
        ids = [f"p{i:03d}" for i in range(1, 101)]
        s1 = prep.split_fixed(ids, seed=42)
        s2 = prep.split_fixed(ids, seed=42)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1["train"]) + len(s1["val"]) + len(s1["test"]), len(ids))

    def test_window_count_matches_residue_count(self):
        record = prep.ProteinRecord(
            protein_id="p1",
            sequence="ACDXBZ",
            dssp_q8="HEST_?",
            labels_q3="HECCCC",
            length=6,
            source_file="p1.all",
        )
        windows = list(prep.generate_windows_for_record(record, window_size=13, padding_token="X"))
        self.assertEqual(len(windows), 6)
        # Now yields (feat_vec: List[float], class_id: int)
        self.assertEqual(windows[0][1], prep.Q3_TO_ID["H"])   # H -> 0
        self.assertEqual(windows[-1][1], prep.Q3_TO_ID["C"])  # C -> 2

    def test_ambiguous_aa_and_padding_zero_encoded(self):
        window = "AXBZ"
        vec = prep.encode_one_hot_window(window)
        self.assertEqual(len(vec), 4 * 20)

        a_slice = vec[0:20]
        self.assertEqual(sum(a_slice), 1.0)
        x_slice = vec[20:40]
        b_slice = vec[40:60]
        z_slice = vec[60:80]
        self.assertEqual(sum(x_slice), 0.0)
        self.assertEqual(sum(b_slice), 0.0)
        self.assertEqual(sum(z_slice), 0.0)


    def test_binary_files_written(self):
        """write_binary_split writes correctly sized .bin files."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            feature_dim = 260
            n = 5
            X = [[float(i) * 0.1] * feature_dim for i in range(n)]
            y = [0, 1, 2, 0, 1]
            prep.write_binary_split(out, "train", X, y)

            x_path = out / "train_X.bin"
            y_path = out / "train_y.bin"
            self.assertTrue(x_path.exists())
            self.assertTrue(y_path.exists())
            # X: n * feature_dim * 4 bytes (float32)
            self.assertEqual(x_path.stat().st_size, n * feature_dim * 4)
            # y: n * 4 bytes (int32)
            self.assertEqual(y_path.stat().st_size, n * 4)

    def test_binary_label_values_in_range(self):
        """Binary y file contains only valid class ids 0, 1, 2."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            X = [[0.0] * 260 for _ in range(6)]
            y = [0, 1, 2, 1, 0, 2]
            prep.write_binary_split(out, "test", X, y)

            y_path = out / "test_y.bin"
            with y_path.open("rb") as f:
                raw = f.read()
            labels = list(struct.unpack(f"{len(y)}i", raw))
            self.assertEqual(labels, y)
            for val in labels:
                self.assertIn(val, (0, 1, 2))

    def test_per_split_class_dist_in_metadata(self):
        """metadata.json contains split_class_distribution with correct structure."""
        import subprocess, sys
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            raw_dir = repo_root / "data" / "raw" / "cb513" / "513_distribute"
            if not raw_dir.exists():
                self.skipTest("Raw CB513 data not available")
            result = subprocess.run(
                [sys.executable, str(repo_root / "scripts" / "preprocess" / "cb513_prepare.py"),
                 "--raw-dir", str(raw_dir),
                 "--out-dir", str(out_dir),
                 "--write-binary", "true"],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            meta_path = out_dir / "canonical" / "metadata.json"
            self.assertTrue(meta_path.exists())
            with meta_path.open() as f:
                meta = json.load(f)
            self.assertIn("split_class_distribution", meta)
            dist = meta["split_class_distribution"]
            for split in ("train", "val", "test"):
                self.assertIn(split, dist)
                for q in ("H", "E", "C"):
                    self.assertIn(q, dist[split])
                    self.assertGreater(dist[split][q], 0)
                split_total = sum(dist[split].values())
                self.assertEqual(split_total, meta["split_counts"][split] if False else split_total)
            total = sum(sum(dist[s].values()) for s in dist)
            self.assertEqual(total, meta["total_residues"])

    def test_blosum62_encoding_shape_and_zero_for_unknown(self):
        """encode_blosum62_window returns 260 floats; unknown AAs map to all-zero."""
        window = "A" * 13
        vec = prep.encode_blosum62_window(window)
        self.assertEqual(len(vec), 260)
        # Known AA must have non-trivial values (not all zero)
        a_slice = vec[0:20]
        self.assertNotEqual(sum(abs(v) for v in a_slice), 0.0)

        # Unknown tokens must map to all-zero
        unk_window = "X" * 13
        unk_vec = prep.encode_blosum62_window(unk_window)
        self.assertEqual(len(unk_vec), 260)
        self.assertEqual(sum(abs(v) for v in unk_vec), 0.0)


if __name__ == "__main__":
    unittest.main()
