import importlib.util
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
        self.assertEqual(windows[0][1], "H")
        self.assertEqual(windows[-1][1], "C")

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


if __name__ == "__main__":
    unittest.main()
