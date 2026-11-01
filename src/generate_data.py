"""Study-level dataset generated from known parameters: effect of agricultural inhibitors on soil N2O emissions and crop yield.

Generated with a fixed seed from known parameters (plausible field-trial designs); the TRUE effects are known,
so we can check that the meta-analysis recovers them. Small studies with non-significant N2O
reductions are partly 'unpublished' to emulate publication bias.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 2025
TRUE_N2O = {"Nitrification inhibitor": -0.45, "Urease inhibitor": -0.12, "Double inhibitor": -0.50}
TRUE_YIELD = {"Nitrification inhibitor": 0.03, "Urease inhibitor": 0.04, "Double inhibitor": 0.05}
NRATE_SLOPE = -0.0010     # lnRR per kg N/ha above 150 (stronger N2O cut at high N)
TAU_N2O, TAU_YIELD = 0.18, 0.05


def generate(seed: int = SEED, n_studies: int = 46) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows, unpublished = [], 0
    for s in range(1, n_studies + 1):
        year = int(rng.integers(2006, 2026))
        crop = rng.choice(["Maize", "Wheat", "Rice", "Vegetables"], p=[.35, .3, .2, .15])
        climate = rng.choice(["Arid", "Temperate", "Humid subtropical"], p=[.25, .45, .3])
        soil = rng.choice(["Coarse", "Medium", "Fine"], p=[.3, .45, .25])
        reps = int(rng.choice([3, 4, 5, 6], p=[.35, .35, .2, .1]))
        n_obs = int(rng.integers(1, 4))
        for o in range(n_obs):
            inh = rng.choice(list(TRUE_N2O), p=[.5, .3, .2])
            nrate = float(rng.choice([90, 120, 150, 180, 220, 260, 300]))
            mu = TRUE_N2O[inh] + NRATE_SLOPE * (nrate - 150) + rng.normal(0, TAU_N2O)
            ctrl = rng.lognormal(np.log(1.5 + 0.012 * nrate), 0.4)          # kg N2O-N/ha
            cv = rng.uniform(0.15, 0.45)
            m_c = np.mean(rng.normal(ctrl, cv * ctrl, reps).clip(0.05))
            m_t = np.mean(rng.normal(ctrl * np.exp(mu), cv * ctrl * np.exp(mu), reps).clip(0.02))
            sd_c, sd_t = cv * m_c * rng.uniform(.8, 1.2), cv * m_t * rng.uniform(.8, 1.2)
            ymu = TRUE_YIELD[inh] + rng.normal(0, TAU_YIELD)
            yc = rng.normal({"Maize": 9.5, "Wheat": 6.0, "Rice": 7.5, "Vegetables": 30}[crop], 1.0)
            ycv = rng.uniform(0.05, 0.12)
            yt = yc * np.exp(ymu) * rng.normal(1, ycv / np.sqrt(reps))
            # publication filter: small, non-significant N2O reductions less likely to be published
            lnrr = np.log(m_t / m_c)
            se = np.sqrt(sd_t**2 / (reps * m_t**2) + sd_c**2 / (reps * m_c**2))
            if lnrr / se > -1.96 and reps <= 4 and rng.random() < 0.6:
                unpublished += 1
                continue
            rows.append(dict(study_id=f"S{s:02d}", obs_id=f"S{s:02d}-{o+1}", year=year, crop=crop, climate=climate,
                             soil_texture=soil, inhibitor=inh, n_rate=nrate, reps=reps,
                             n2o_ctrl_mean=m_c, n2o_ctrl_sd=sd_c, n2o_trt_mean=m_t, n2o_trt_sd=sd_t,
                             yield_ctrl_mean=yc, yield_ctrl_sd=yc * ycv, yield_trt_mean=yt, yield_trt_sd=yt * ycv))
    df = pd.DataFrame(rows)
    meta = {"generated_observations": len(df) + unpublished, "unpublished_removed": unpublished,
            "true_n2o_lnRR": TRUE_N2O, "true_yield_lnRR": TRUE_YIELD, "true_nrate_slope": NRATE_SLOPE}
    return df, meta


if __name__ == "__main__":
    d, m = generate()
    d.to_csv(ROOT / "data" / "studies.csv", index=False)
    print(len(d), d.study_id.nunique(), m)
