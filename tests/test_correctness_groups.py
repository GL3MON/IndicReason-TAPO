import unittest
from train_tapo_trl import correctness_group_metrics

class CorrectnessGroupsTests(unittest.TestCase):
    def test_all_wrong_all_correct_and_mixed_groups(self):
        metrics = correctness_group_metrics([0]*8 + [1]*8 + [1,0,0,0,0,0,0,0], 8)
        self.assertAlmostEqual(metrics['tapo/correctness_all_wrong_fraction'], 1/3)
        self.assertAlmostEqual(metrics['tapo/correctness_all_correct_fraction'], 1/3)
        self.assertAlmostEqual(metrics['tapo/correctness_mixed_fraction'], 1/3)
        self.assertAlmostEqual(metrics['tapo/correctness_any_correct_fraction'], 2/3)
        self.assertEqual(metrics['tapo/correct_completions_per_group'], 3)

    def test_rejects_partial_nonbinary_or_empty_groups(self):
        for values, size in (([],8), ([0]*7,8), ([0.5]*8,8), ([0],1)):
            with self.assertRaises(ValueError):
                correctness_group_metrics(values,size)

if __name__ == '__main__':
    unittest.main()
