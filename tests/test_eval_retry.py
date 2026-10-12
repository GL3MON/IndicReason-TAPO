import unittest
from evaluate_qwen35_sft import select_failed_rows

class RetryTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(source_id=f'gsm8k_test_{i}',question=f'ta{i}',english_question=f'en{i}',answer=str(i)) for i in range(3)]
        self.previous=[dict(r,correct=(i==1)) for i,r in enumerate(self.rows)]
    def test_only_failed_questions_without_prior_completions_or_gold_changes(self):
        self.assertEqual(select_failed_rows(self.rows,self.previous),[self.rows[0],self.rows[2]])
    def test_rejects_changed_missing_and_incomplete_records(self):
        bad=[dict(r) for r in self.previous];bad[0]['answer']='999'
        for previous in (self.previous[:-1],self.previous[::-1],bad):
            with self.assertRaises(ValueError):select_failed_rows(self.rows,previous)

if __name__=='__main__':unittest.main()
