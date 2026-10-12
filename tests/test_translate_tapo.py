import json
import tempfile
import unittest

from indic_reasoning.translation.translate_tapo import encode_row, has_target_script, recover_output


class ResumeTests(unittest.TestCase):
    def test_target_script_distinguishes_malayalam_and_tamil(self):
        self.assertTrue(has_target_script("മലയാളം", "ml"))
        self.assertFalse(has_target_script("தமிழ்", "ml"))
        self.assertFalse(has_target_script("English", "ml"))
        self.assertTrue(has_target_script("தமிழ்", "ta"))
        self.assertFalse(has_target_script("മലയാളം", "ta"))
        self.assertFalse(has_target_script("", "ml"))

    def test_unicode_line_separators_remain_one_tapo_line(self):
        row = {"question": "தமிழ்\u0085\u2028\u2029"}
        encoded = encode_row(row).decode("utf-8")
        self.assertEqual(len(encoded.splitlines()), 1)
        self.assertEqual(json.loads(encoded), row)

    def setUp(self):
        self.dataset = [{"question": "How many?", "answer": "Reasoning\n#### 42"}]
        self.row = dict(question="எத்தனை?", english_question="How many?",
                        answer="42", language="ta", source_id="gsm8k_train_0")

    def test_recovers_partial_tail_without_losing_saved_row(self):
        saved = (json.dumps(self.row, ensure_ascii=False) + "\n").encode()
        with tempfile.TemporaryFile("w+b") as handle:
            handle.write(saved + b'{"question":')
            self.assertEqual(recover_output(handle, self.dataset, "ta"), {0})
            handle.seek(0)
            self.assertEqual(handle.read(), saved)

    def test_rejects_duplicate_and_misaligned_rows(self):
        for rows in ([self.row, self.row], [dict(self.row, answer="99")],
                     [dict(self.row, english_question="Another question")]):
            with self.subTest(rows=rows), tempfile.TemporaryFile("w+b") as handle:
                handle.write("".join(json.dumps(row) + "\n" for row in rows).encode())
                with self.assertRaises(ValueError):
                    recover_output(handle, self.dataset, "ta")
