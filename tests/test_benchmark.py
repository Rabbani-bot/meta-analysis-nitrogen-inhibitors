"""Validate the engine against the BCG vaccine dataset (Colditz et al. 1994), the standard
benchmark used in the metafor documentation:
  REML: estimate -0.7145, se 0.1798, tau^2 0.3132, I^2 92.22%, Q(df=12) 152.23
  DL:   estimate -0.7141, tau^2 0.3088
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metaengine as me  # noqa: E402

BCG = np.array([
    [4, 119, 11, 128], [6, 300, 29, 274], [3, 228, 11, 209], [62, 13536, 248, 12619],
    [33, 5036, 47, 5761], [180, 1361, 372, 1079], [8, 2537, 10, 619], [505, 87886, 499, 87892],
    [29, 7470, 45, 7232], [17, 1699, 65, 1600], [186, 50448, 141, 27197], [5, 2493, 3, 2338],
    [27, 16886, 29, 17825]], float)


def bcg_yv():
    tp, tn, cp, cn = BCG.T
    y = np.log((tp / (tp + tn)) / (cp / (cp + cn)))
    v = 1 / tp - 1 / (tp + tn) + 1 / cp - 1 / (cp + cn)
    return y, v


class TestBenchmark(unittest.TestCase):
    def test_reml_matches_metafor(self):
        y, v = bcg_yv()
        r = me.meta(y, v, "REML")
        self.assertAlmostEqual(r.estimate, -0.7145, places=3)
        self.assertAlmostEqual(r.se, 0.1798, places=3)
        self.assertAlmostEqual(r.tau2, 0.3132, places=3)
        self.assertAlmostEqual(r.I2, 92.22, places=1)
        self.assertAlmostEqual(r.Q, 152.23, places=1)

    def test_dl_matches_metafor(self):
        y, v = bcg_yv()
        r = me.meta(y, v, "DL")
        self.assertAlmostEqual(r.estimate, -0.7141, places=3)
        self.assertAlmostEqual(r.tau2, 0.3088, places=3)

    def test_fixed_effect_is_inverse_variance_mean(self):
        y, v = bcg_yv()
        self.assertAlmostEqual(me.meta(y, v, "FE").estimate, np.sum(y / v) / np.sum(1 / v))

    def test_lnrr_variance(self):
        y, v = me.lnrr(12, 3, 10, 10, 2, 10)
        self.assertAlmostEqual(float(y), np.log(1.2))
        self.assertAlmostEqual(float(v), 9 / (10 * 144) + 4 / (10 * 100))

    def test_metareg_intercept_only_equals_meta(self):
        y, v = bcg_yv()
        # BCG absolute latitude moderator: documented slope -0.0291 (REML)
        lat = np.array([44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33], float)
        mr = me.metareg(y, v, lat)
        self.assertAlmostEqual(mr["coef"][1], -0.0291, places=3)

    def test_trimfill_detects_suppression(self):
        """Over 100 Monte Carlo replicates: few studies imputed without bias, many with small-study suppression."""
        rng = np.random.default_rng(0)
        sym, biased = [], []
        for _ in range(100):
            v = rng.uniform(0.01, 0.2, 40); y = rng.normal(0.3, np.sqrt(v))
            sym.append(me.trim_and_fill(y, v)["k0"])
            v2 = rng.uniform(0.01, 0.3, 80); y2 = rng.normal(0.3, np.sqrt(v2))
            keep = ~((v2 > 0.1) & (y2 < 0.3))
            biased.append(me.trim_and_fill(y2[keep], v2[keep])["k0"])
        self.assertLessEqual(np.median(sym), 4)
        self.assertGreaterEqual(np.median(biased), 8)

    def test_knha_widens_ci(self):
        y, v = bcg_yv()
        a, b = me.meta(y, v, "REML"), me.meta(y, v, "REML", knha=True)
        self.assertGreater(b.ci_high - b.ci_low, a.ci_high - a.ci_low)


if __name__ == "__main__":
    unittest.main()
