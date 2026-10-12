import json
from pathlib import Path
import tempfile
import unittest
from train_tapo_trl import read_jsonl

class JSONLRecordsTests(unittest.TestCase):
    def test_embedded_unicode_separators_remain_inside_question(self):
        rows=[{'question':'தமிழ்\u2028English\u2029next\x85line','answer':'4'}, {'question':'ordinary','answer':'5'}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            path.write_text('\r\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\r\n\r\n',encoding='utf-8')
            self.assertEqual(read_jsonl(path),rows)

if __name__=='__main__':
    unittest.main()
