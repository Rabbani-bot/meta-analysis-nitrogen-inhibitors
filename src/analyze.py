"""Full meta-analysis workflow and figures."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT.parent / "common"))
import metaengine as me  # noqa: E402
import style  # noqa: E402
from generate_data import generate  # noqa: E402

FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "outputs"
pct = lambda x: 100 * (np.exp(x) - 1)
ORDER = ["Nitrification inhibitor", "Urease inhibitor", "Double inhibitor"]
COL = dict(zip(ORDER, [style.PALETTE[0], style.PALETTE[2], style.PALETTE[1]]))


def main() -> dict:
    style.apply()
    df, truth = generate()
    df.to_csv(ROOT / "data" / "studies.csv", index=False)
    df["yi"], df["vi"] = me.lnrr(df.n2o_trt_mean, df.n2o_trt_sd, df.reps, df.n2o_ctrl_mean, df.n2o_ctrl_sd, df.reps)
    df["yi_y"], df["vi_y"] = me.lnrr(df.yield_trt_mean, df.yield_trt_sd, df.reps, df.yield_ctrl_mean, df.yield_ctrl_sd, df.reps)
    df.to_csv(ROOT / "data" / "effect_sizes.csv", index=False)
    y, v = df.yi.to_numpy(), df.vi.to_numpy()
    res = {"k_obs": len(df), "k_studies": int(df.study_id.nunique()), "truth": truth}

    overall = {m: me.meta(y, v, m) for m in ["FE", "DL", "REML"]}
    overall["REML+KnHa"] = me.meta(y, v, "REML", knha=True)
    res["overall_n2o"] = {k: {**r.as_dict(), **r.pct()} for k, r in overall.items()}
    yld = me.meta(df.yi_y, df.vi_y, "REML", knha=True)
    res["overall_yield"] = {**yld.as_dict(), **yld.pct()}
    # robustness to non-independence: one aggregated effect per study
    agg = df.groupby("study_id").apply(lambda g: pd.Series({"yi": np.average(g.yi, weights=1 / g.vi),
                                                            "vi": 1 / np.sum(1 / g.vi)}), include_groups=False)
    res["study_level_n2o"] = {**me.meta(agg.yi, agg.vi, "REML", knha=True).as_dict()}

    per, qb = me.subgroups(y, v, df.inhibitor.to_numpy())
    per_y, qb_y = me.subgroups(df.yi_y.to_numpy(), df.vi_y.to_numpy(), df.inhibitor.to_numpy())
    rows = []
    for g in ORDER:
        r, ry = per[g], per_y[g]
        rows.append({"inhibitor": g, "k": r.k, "n2o_pct": pct(r.estimate), "n2o_ci": (pct(r.ci_low), pct(r.ci_high)),
                     "n2o_true_pct": pct(truth["true_n2o_lnRR"][g] + 0.0), "I2": r.I2,
                     "yield_pct": pct(ry.estimate), "yield_ci": (pct(ry.ci_low), pct(ry.ci_high))})
    res["subgroups"] = {"table": rows, "Q_between_n2o": qb, "Q_between_yield": qb_y}
    for factor in ["crop", "climate", "soil_texture"]:
        _, q = me.subgroups(y, v, df[factor].to_numpy())
        res[f"moderator_{factor}"] = q

    X = np.column_stack([df.n_rate - 150, pd.get_dummies(df.inhibitor)[["Urease inhibitor", "Double inhibitor"]].astype(float)])
    mr = me.metareg(y, v, X)
    res["metareg"] = {"terms": ["intercept (NI at 150 kg N)", "N rate (per kg above 150)", "UI vs NI", "Double vs NI"],
                      "coef": mr["coef"].tolist(), "se": mr["se"].tolist(), "p": mr["p"].tolist(),
                      "R2_pct": mr["R2_pct"], "tau2_resid": mr["tau2_resid"], "QE": mr["QE"], "QE_p": mr["QE_p"]}

    eg = me.egger(y, v)
    tf = me.trim_and_fill(y, v)
    res["bias"] = {"egger": eg, "trimfill_k0": tf["k0"], "trimfill_side": tf["side"],
                   "adjusted_pct": pct(tf["adjusted"].estimate), "failsafe_N": me.failsafe_n(y, v)}
    lo, hi, boots = me.bootstrap_ci(y, v, B=2000)
    res["bootstrap_bc_ci_pct"] = [pct(lo), pct(hi)]
    loo = me.leave_one_out(y, v)
    res["leave_one_out_range_pct"] = [pct(min(r.estimate for r in loo)), pct(max(r.estimate for r in loo))]
    cum, idx = me.cumulative(y, v, df.year.to_numpy())

    # ---------- tables ----------
    t1 = pd.DataFrame([{"model": k, "k": r.k, "lnRR": r.estimate, "SE": r.se, "% change": pct(r.estimate),
                        "95% CI": f"{pct(r.ci_low):.1f} to {pct(r.ci_high):.1f}",
                        "95% PI": f"{pct(r.pi_low):.1f} to {pct(r.pi_high):.1f}",
                        "tau2": r.tau2, "I2 %": r.I2, "Q": r.Q, "p(Q)": r.Q_p} for k, r in overall.items()])
    t1.round(4).to_csv(OUT / "table1_overall_models.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "table2_subgroups.csv", index=False)
    pd.DataFrame({k: res["metareg"][k] for k in ["terms", "coef", "se", "p"]}).to_csv(OUT / "table3_metaregression.csv", index=False)

    # ---------- figures ----------
    fig_orchard(df, per, overall["REML+KnHa"])
    fig_forest(df, per, per_y, overall["REML+KnHa"], yld)
    fig_funnel(y, v, tf, overall["REML"])
    fig_bubble(df, mr)
    fig_cumulative(df, cum, idx)
    fig_influence(df, loo, overall["REML"], boots)
    fig_winwin(df)
    fig_moderators(df, y, v)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    return res


def fig_orchard(df, per, ov):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rng = np.random.default_rng(3)
    for i, g in enumerate(ORDER):
        d = df[df.inhibitor == g]
        jit = rng.uniform(-0.28, 0.28, len(d))
        ax.scatter(pct(d.yi), i + jit, s=18 + 600 * (1 / d.vi) / (1 / df.vi).max(), color=COL[g], alpha=0.35, edgecolor="white", lw=0.5)
        r = per[g]
        ax.plot([pct(r.pi_low), pct(r.pi_high)], [i, i], color="#222", lw=1.2)
        ax.plot([pct(r.ci_low), pct(r.ci_high)], [i, i], color="#222", lw=5, solid_capstyle="round")
        ax.scatter(pct(r.estimate), i, s=110, color="white", edgecolor="#222", zorder=5, lw=1.5)
        ax.text(105, i, f"k = {r.k}\n{pct(r.estimate):.1f}% [{pct(r.ci_low):.1f}, {pct(r.ci_high):.1f}]", va="center", fontsize=9)
    ax.axvline(0, color=style.NEUTRAL, ls="--", lw=1)
    ax.set_yticks(range(3)); ax.set_yticklabels(ORDER); ax.set_xlim(-90, 100)
    ax.set_xlabel("Change in N2O emissions vs control (%)")
    ax.set_title("Orchard plot: individual effects (size = precision), pooled mean, 95% CI (thick) and prediction interval (thin)")
    style.footnote(fig); fig.savefig(FIG / "01_orchard_plot.png"); plt.close(fig)


def fig_forest(df, per, per_y, ov, yld):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    labels = ORDER + ["Overall"]
    for ax, (res, ovr, title, xl) in zip(axes, [(per, ov, "N2O emissions", (-70, 10)), (per_y, yld, "Crop yield", (-5, 12))]):
        for i, g in enumerate(labels):
            r = ovr if g == "Overall" else res[g]
            c = "#222" if g == "Overall" else COL[g]
            ax.plot([pct(r.ci_low), pct(r.ci_high)], [i, i], color=c, lw=2.5)
            ax.scatter(pct(r.estimate), i, marker="D" if g == "Overall" else "s", s=80, color=c, zorder=4)
            ax.text(xl[1], i, f"  {pct(r.estimate):+.1f}%", va="center", fontsize=9)
        ax.axvline(0, color=style.NEUTRAL, ls="--", lw=1); ax.set_xlim(*xl)
        ax.set_title(title); ax.set_xlabel("% change vs control (95% CI)")
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
    fig.suptitle("Forest plot by inhibitor type: emissions fall while yields hold", x=0.01, ha="left", fontweight="semibold")
    style.footnote(fig); fig.tight_layout(); fig.savefig(FIG / "02_forest_by_inhibitor.png"); plt.close(fig)


def fig_funnel(y, v, tf, ov):
    se = np.sqrt(v)
    fig, ax = plt.subplots(figsize=(8, 6))
    smax = se.max() * 1.1
    s_grid = np.linspace(0, smax, 50)
    for z, shade in [(2.576, "#F1F3F5"), (1.96, "#E3E7EB"), (1.645, "#D3D9DF")]:
        ax.fill_betweenx(s_grid, -z * s_grid, z * s_grid, color=shade, lw=0)
    ax.plot(ov.estimate - 1.96 * s_grid, s_grid, color=style.NEUTRAL, lw=0.8)
    ax.plot(ov.estimate + 1.96 * s_grid, s_grid, color=style.NEUTRAL, lw=0.8)
    ax.scatter(y, se, color=style.PALETTE[0], s=28, label="published effects", zorder=3)
    if tf["k0"]:
        ax.scatter(tf["filled_y"], np.sqrt(tf["filled_v"]), facecolor="white", edgecolor=style.PALETTE[3], s=28,
                   label=f"imputed by trim-and-fill (k0 = {tf['k0']})", zorder=3)
    ax.axvline(ov.estimate, color=style.PALETTE[0], lw=1.2, label=f"REML mean ({pct(ov.estimate):.1f}%)")
    ax.axvline(tf["adjusted"].estimate, color=style.PALETTE[3], lw=1.2, ls="--",
               label=f"bias-adjusted ({pct(tf['adjusted'].estimate):.1f}%)")
    ax.invert_yaxis(); ax.set_xlabel("lnRR (N2O)"); ax.set_ylabel("Standard error")
    ax.set_title("Contour-enhanced funnel plot with trim-and-fill\nshaded: p > 0.10, 0.05, 0.01 regions around zero")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5); style.footnote(fig); fig.savefig(FIG / "03_funnel_trimfill.png"); plt.close(fig)


def fig_bubble(df, mr):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g in ORDER:
        d = df[df.inhibitor == g]
        ax.scatter(d.n_rate, pct(d.yi), s=900 * (1 / d.vi) / (1 / df.vi).max() + 10, color=COL[g], alpha=0.5, edgecolor="white", label=g)
    xs = np.linspace(80, 310, 100)
    b = mr["coef"]
    ax.plot(xs, pct(b[0] + b[1] * (xs - 150)), color="#222", lw=2, label=f"meta-regression (NI): slope {b[1]*100:.2f} lnRR per 100 kg N")
    ax.axhline(0, color=style.NEUTRAL, ls="--", lw=1)
    ax.set_xlabel("Nitrogen application rate (kg N/ha)"); ax.set_ylabel("Change in N2O (%)")
    ax.set_title(f"Meta-regression: larger N2O cuts at higher N rates (R2 = {mr['R2_pct']:.0f}% of heterogeneity explained)")
    ax.legend(fontsize=8.5, loc="upper right"); style.footnote(fig); fig.savefig(FIG / "04_metaregression_bubble.png"); plt.close(fig)


def fig_cumulative(df, cum, idx):
    years = df.year.to_numpy()[idx][1:]
    est = np.array([pct(r.estimate) for r in cum]); lo = np.array([pct(r.ci_low) for r in cum]); hi = np.array([pct(r.ci_high) for r in cum])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(est)) + 2
    ax.fill_between(x, lo, hi, color=style.PALETTE[1], alpha=0.25, lw=0)
    ax.plot(x, est, color=style.PALETTE[1], lw=2)
    ax.axhline(0, color=style.NEUTRAL, ls="--", lw=1)
    for i in range(0, len(x), 8):
        ax.annotate(str(years[i]), (x[i], hi[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#666")
    ax.set_xlabel("Number of observations accumulated (in order of publication year)"); ax.set_ylabel("Pooled N2O change (%)")
    ax.set_title("Cumulative meta-analysis: the effect stabilises early and the CI narrows")
    style.footnote(fig); fig.savefig(FIG / "05_cumulative.png"); plt.close(fig)


def fig_influence(df, loo, ov, boots):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    e = np.array([pct(r.estimate) for r in loo])
    ax[0].scatter(range(len(e)), e, s=14, color=style.PALETTE[0])
    ax[0].axhline(pct(ov.estimate), color="#222", lw=1)
    ax[0].fill_between([0, len(e)], pct(ov.ci_low), pct(ov.ci_high), color="#999", alpha=0.15)
    ax[0].set_xlabel("Observation omitted"); ax[0].set_ylabel("Pooled N2O change (%)")
    ax[0].set_title("Leave-one-out: no single observation drives the result")
    ax[1].hist(pct(boots), bins=40, color=style.PALETTE[1], alpha=0.8)
    ax[1].axvline(pct(ov.estimate), color="#222")
    ax[1].set_title("Bootstrap distribution (2,000 resamples)"); ax[1].set_xlabel("Pooled N2O change (%)")
    style.footnote(fig); fig.tight_layout(); fig.savefig(FIG / "06_influence_bootstrap.png"); plt.close(fig)


def fig_winwin(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.axhline(0, color=style.NEUTRAL, lw=1); ax.axvline(0, color=style.NEUTRAL, lw=1)
    ax.fill_between([-100, 0], 0, 30, color="#8AC926", alpha=0.08)
    ax.text(-85, 22, "WIN-WIN\nlower N2O, higher yield", color="#3B7A00", fontweight="semibold", fontsize=9)
    for g in ORDER:
        d = df[df.inhibitor == g]
        ax.scatter(pct(d.yi), pct(d.yi_y), color=COL[g], s=35, alpha=0.75, label=g, edgecolor="white")
    share = np.mean((df.yi < 0) & (df.yi_y > 0)) * 100
    ax.set_xlim(-90, 60); ax.set_ylim(-15, 30)
    ax.set_xlabel("Change in N2O (%)"); ax.set_ylabel("Change in yield (%)")
    ax.set_title(f"Trade-off space: {share:.0f}% of observations are win-win")
    ax.legend(fontsize=8.5, loc="lower right"); style.footnote(fig); fig.savefig(FIG / "07_winwin_quadrant.png"); plt.close(fig)


def fig_moderators(df, y, v):
    rows = []
    for f in ["crop", "climate", "soil_texture"]:
        per, q = me.subgroups(y, v, df[f].to_numpy())
        for g, r in per.items():
            rows.append((f.replace("_", " ").title(), g, r.k, pct(r.estimate), pct(r.ci_low), pct(r.ci_high), q["p"]))
    t = pd.DataFrame(rows, columns=["moderator", "level", "k", "est", "lo", "hi", "p_between"])
    t.to_csv(OUT / "table4_moderators.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ypos, labels = [], []
    yy = 0
    for mname, g in t.groupby("moderator", sort=False):
        ax.text(-74, yy - 0.15, f"{mname}  (Q_between p = {g.p_between.iloc[0]:.2f})", fontweight="semibold", fontsize=9.5)
        yy += 1
        for _, r in g.iterrows():
            ax.plot([r.lo, r.hi], [yy, yy], color=style.PALETTE[0], lw=2.2)
            ax.scatter(r.est, yy, color=style.PALETTE[0], s=40, zorder=3)
            labels.append(f"{r.level} (k={r.k})"); ypos.append(yy); yy += 1
        yy += 0.4
    ax.set_yticks(ypos); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.axvline(0, color=style.NEUTRAL, ls="--", lw=1); ax.set_xlim(-75, 10)
    ax.set_xlabel("Change in N2O (%) with 95% CI"); ax.set_title("Subgroup analysis by crop, climate and soil texture")
    style.footnote(fig); fig.savefig(FIG / "08_moderators.png"); plt.close(fig)


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ["k_obs", "k_studies", "subgroups", "bias", "bootstrap_bc_ci_pct",
                                         "leave_one_out_range_pct"]}, indent=1, default=str)[:3500])
    print("overall REML+KnHa", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["overall_n2o"]["REML+KnHa"].items()})
    print("yield", round(r["overall_yield"]["estimate_pct"], 2), "metareg", r["metareg"]["coef"], r["metareg"]["p"], r["metareg"]["R2_pct"])
    print("study-level", round(100*(np.exp(r["study_level_n2o"]["estimate"])-1),2))
