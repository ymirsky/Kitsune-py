import unittest

import numpy as np

from KitNET.KitNET import KitNET


class TestBatchScores(unittest.TestCase):
    def test_batch_matches_stream(self):
        rng = np.random.RandomState(7)
        model = KitNET(8, max_autoencoder_size=3, FM_grace_period=10, AD_grace_period=40)
        for row in rng.rand(51, 8):
            model.process(row)
        rows = rng.rand(24, 8)
        expected = [model.process(row) for row in rows]
        for batch in (rows, rows[:1], rows[:0]):
            np.testing.assert_allclose(model.execute(batch), expected[:len(batch)], rtol=1e-12)
        self.assertEqual(model.n_executed, 24 + 24 + 1)
        self.assertEqual(model.n_trained, 51)
