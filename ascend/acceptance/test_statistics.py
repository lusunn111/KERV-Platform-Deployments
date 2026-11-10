import unittest
from acceptance.aggregate import stats, acceptance_improvement_pass, success_loss_pass

class StatisticsBoundaries(unittest.TestCase):
    def test_zero_acceptance_kept(self):
        self.assertEqual(stats([0,2,4])['mean'],2)
        self.assertEqual(stats([0,0])['count'],2)
    def test_nearest_rank(self):
        self.assertEqual(stats(list(range(1,21)))['p95_nearest_rank'],19)
    def test_strict_acceptance(self):
        self.assertFalse(acceptance_improvement_pass(.5));self.assertTrue(acceptance_improvement_pass(.50001))
    def test_strict_success_loss(self):
        self.assertFalse(success_loss_pass(100,100,95,100));self.assertTrue(success_loss_pass(100,100,96,100))
    def test_nonfinite(self):
        for value in [float('nan'),float('inf'),float('-inf')]:
            with self.assertRaises(ValueError):stats([value])

if __name__=='__main__':unittest.main()
