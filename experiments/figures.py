"""Paper figures from the run artifacts (no new experiments).

  .venv/Scripts/python.exe -m experiments.figures            # KD_PROFILE picks dev / full
  .venv/Scripts/python.exe -m experiments.figures --only yield detection

Writes docs/figures/<profile>/<name>.pdf (vector, for the paper) and .png (300 dpi), plus manifest.json recording the
source artifact and its versions for every figure. A figure whose inputs do not exist yet is skipped with
the reason (the scaling curve and DCM need the GPU steps; search needs experiments.search runs).

Colors: validated categorical slots 1-3 (blue, orange, aqua; all-pairs CVD-safe, light surface). Every
series also has its own marker or line style, so figures survive grayscale print. Text stays in text ink.
"""
import argparse
import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from configs import ROOT, load  # noqa: E402
from experiments.run import BOUNDS, CANARY, TAUS  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
RAMP = ["#86b6ef", "#2a78d6", "#104281"]               # ordinal blue (steps 250 / 450 / 650)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
CONFIGS = [("none_plain", "No defense"), ("l0_plain", "Quotas only"), ("mmd_plain", "Per-account MMD"),
           ("seat_plain", "SEAT (per account)"), ("simlink_plain", "Similarity linkage"),
           ("none_coupled", "Coupled sampling alone"), ("full_coupled", "Full gateway")]
STAGE = "_benign_eval+attacks_stage"
# color follows the configuration, never its position: a missing run must not repaint the others
STYLE = {"none_plain": (S1, "o", "-", -0.22), "none_coupled": (S2, "s", "--", 0.0), "full_coupled": (S3, "^", ":", 0.22)}
COL, WIDE = 3.4, 7.0  # single- and double-column widths, inches

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.5, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white"})


class Figs:
    def __init__(self, cfg):
        self.cfg, self.runs = cfg, cfg["processed"] / "runs"
        self.out = ROOT / "docs" / "figures" / cfg["profile"]  # dev and full never overwrite each other
        self.manifest = {}

    def report(self, key, tag=""):
        f = self.runs / f"{key}_{self.cfg['profile']}{tag}" / "report.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def reports(self, tag=""):
        return [(lab, r) for k, lab in CONFIGS if (r := self.report(k, tag))]

    def save(self, fig, name, sources, note=""):
        self.out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        self.manifest[name] = {"profile": self.cfg["profile"], "sources": sources, "note": note}
        print(f"  wrote docs/figures/{self.cfg['profile']}/{name}.pdf/.png")


KFMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")  # log axes: 2, 5, 10 instead of 2x10^0


def label_bars(ax, bars, fmt, clear=0.0):
    """Value at the bar end, pushed right of x = clear (a reference line) so the two never collide."""
    for b in bars:
        ax.annotate(fmt(b.get_width()), (max(b.get_width(), clear), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK2)


def versions(r):
    return {**r.get("versions", {}), "policy_thresholds": r.get("policy_thresholds")}


# ---------------------------------------------------------------- figures

def scaling(F):
    """Step-5 data-scaling curve: canary accuracy vs training examples, per seed and mean, with tau lines."""
    f, g = ROOT / "runs" / "scaling" / "results.jsonl", F.cfg["processed"] / "cache" / "gate.json"
    if not (f.exists() and g.exists()):
        return "runs/scaling/results.jsonl or cache/gate.json missing (GPU step 5)"
    model = F.cfg["student"]["curve_model"]
    rows = [r for r in map(json.loads, f.read_text().splitlines()) if r["model"] == model]
    gate = json.loads(g.read_text(encoding="utf-8"))
    if not rows or not gate.get("teacher"):
        return f"no scaling rows for {model} or no gated teacher"
    mean = lambda a: float(np.mean(list(a.values()))) if isinstance(a, dict) else float(a)
    t_acc = mean(gate["results"][gate["teacher"]]["accuracy"])
    by = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["n"]] = mean(r["accuracy"])
    ns = sorted({n for d in by.values() for n in d})
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for s, d in sorted(by.items()):
        x = sorted(d)
        ax.plot(x, [d[n] for n in x], color=MUTED, lw=0.8, marker="o", ms=3, alpha=0.8,
                label="single seed" if s == min(by) else None)
    m = [np.mean([by[s][n] for s in by if n in by[s]]) for n in ns]
    ax.plot(ns, m, color=S1, marker="o", ms=4, label=f"mean of {len(by)} seeds")
    ax.axhline(t_acc, color=INK2, lw=1, ls="--")
    ax.annotate(f"teacher {t_acc:.2f}", (ns[0], t_acc), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    for t in TAUS:
        ax.axhline(t * t_acc, color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"τ = {t:g}", (ns[-1], t * t_acc), xytext=(-2, 3), textcoords="offset points", ha="right",
                    fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Training examples (teacher responses)")
    ax.set_ylabel("Student canary accuracy")
    ax.legend(loc="lower right")
    F.save(fig, "fig_scaling_curve", ["runs/scaling/results.jsonl", "Data/Processed/cache/gate.json"],
           f"student {model}, teacher {gate['teacher']}")


def benign_cost(F):
    """Share of benign evaluation accounts reaching tier 2 and tier 3 per configuration, against the caps."""
    reps = [(l, r) for l, r in F.reports() if r["defense"] not in ("none", "l0")]
    if not reps:
        return "no enforcing runs (mmd / seat / simlink / full)"
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.9), sharey=True)
    y = np.arange(len(reps))
    for ax, key, cap, title in [(axes[0], "share_tier2", 0.01, "Reached tier 2 (Shape)"),
                                (axes[1], "share_tier3", 0.001, "Reached tier 3 (Verify)")]:
        v = [100 * r["benign_cost"][key] for _, r in reps]
        bars = ax.barh(y, v, height=0.6, color=S1)
        label_bars(ax, bars, lambda w: f"{w:.2f}%", clear=100 * cap)
        ax.axvline(100 * cap, color=INK, lw=1, ls="--")
        ax.annotate(f"cap {100 * cap:g}%", (100 * cap, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xlim(0, max(max(v), 100 * cap) * 1.3)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("Benign accounts (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [l for l, _ in reps])
    axes[0].invert_yaxis()
    n = reps[0][1]["benign_cost"]["accounts"]
    F.save(fig, "fig_benign_cost", [f"runs/*_{F.cfg['profile']}/report.json"],
           f"{n} benign evaluation accounts; " + json.dumps({l: versions(r) for l, r in reps[-1:]}))


def yield_bars(F, tag="", name="fig_attacker_yield"):
    """Attacker yield per configuration: unique responses (Q_eff, upper bound) and distinct canary prompts
    (lower bound), with the bootstrap CI over campaigns on Q_eff."""
    reps = F.reports(tag)
    if not reps:
        return f"no reports for streams{tag or ' (default)'}"
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(reps))
    q = np.array([r["attacker"]["Q_eff_total"] for _, r in reps], float)
    d = np.array([sum(c["distinct_canary_prompts"] for c in r["attacker"]["campaigns"].values()) for _, r in reps], float)
    ci = np.array([r["attacker"].get("Q_eff_total_ci95", [v, v]) for (_, r), v in zip(reps, q)], float)
    h = 0.36
    ax.barh(y - h / 2, q, height=h, color=S1, label="Unique responses (Q_eff, hi)")
    ax.errorbar(q, y - h / 2, xerr=[q - ci[:, 0], ci[:, 1] - q], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.barh(y + h / 2, d, height=h, color=S2, hatch="////", edgecolor="white", linewidth=0,
            label="Distinct canary prompts (lo)")
    ax.set_yticks(y, [l for l, _ in reps])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Summed over campaigns")
    ax.xaxis.set_major_formatter(KFMT)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    F.save(fig, name, [f"runs/*_{F.cfg['profile']}{tag}/report.json"], "error bars: 95% bootstrap over campaigns")


def detection(F, tag="", name="fig_detection"):
    """Account- and cluster-level TPR at 1% and 0.1% benign-account FPR with bootstrap 95% CIs."""
    reps = F.reports(tag)
    if not reps:
        return f"no reports for streams{tag or ' (default)'}"
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2), sharey=True)
    y = np.arange(len(reps))
    for ax, k, title in [(axes[0], "tpr@1%fpr", "TPR at 1% benign FPR"), (axes[1], "tpr@0.1%fpr", "TPR at 0.1% benign FPR")]:
        for lvl, col, mk, off in [("account", S1, "o", -0.15), ("cluster", S2, "s", 0.15)]:
            t = np.array([r["detection"][lvl][k]["tpr"] for _, r in reps], float)
            lo = np.array([r["detection"][lvl][k]["ci95"][0] for _, r in reps], float)
            hi = np.array([r["detection"][lvl][k]["ci95"][1] for _, r in reps], float)
            ax.errorbar(t, y + off, xerr=[np.clip(t - lo, 0, None), np.clip(hi - t, 0, None)], fmt=mk, ms=4.5,
                        color=col, ecolor=col, elinewidth=1, capsize=2, label=f"{lvl.capitalize()} level")
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(title, loc="left")
        ax.set_xlabel("True-positive rate (attack accounts)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [l for l, _ in reps])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    d = reps[0][1]["detection"]
    F.save(fig, name, [f"runs/*_{F.cfg['profile']}{tag}/report.json"],
           f"{d['benign_accounts']} benign / {d['attack_accounts']} attack accounts; 500 bootstrap resamples")


def campaigns(F, tag="", name="fig_campaign_yield"):
    """Q_eff per campaign: no defense vs coupled sampling alone vs full gateway, enforced campaigns marked."""
    reps = [(k, l, r) for k, l in CONFIGS if k in STYLE and (r := F.report(k, tag))]
    if len(reps) < 2:
        return "needs at least two of none_plain / none_coupled / full_coupled"
    cids = sorted(reps[0][2]["attacker"]["campaigns"])
    att = {c: reps[0][2]["attacker"]["campaigns"][c]["attack"] for c in cids}
    fig, ax = plt.subplots(figsize=(COL, 0.22 * len(cids) + 0.9))
    y = np.arange(len(cids))
    for k, lab, r in reps:
        col, mk, _, off = STYLE[k]
        ax.scatter([r["attacker"]["campaigns"][c]["Q_eff"] for c in cids], y + off, s=20, color=col, marker=mk,
                   label=lab, zorder=3, edgecolors="white", linewidths=0.6)
    full = reps[-1][2]["attacker"]["campaigns"]
    ax.set_yticks(y, [f"{c} {att[c]}{' *' if full[c]['enforced'] else ''}" for c in cids])
    ax.invert_yaxis()
    ax.set_xlabel("Q_eff (unique full-quality canary responses)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    F.save(fig, name, [f"runs/*_{F.cfg['profile']}{tag}/report.json"], "* = enforced under the full gateway")


def yield_time(F):
    """Cumulative attacker yield over simulated days under three configurations."""
    P, T = F.cfg["processed"], F.cfg["processed"] / "traffic"
    gt = [(T / s / "ground_truth.parquet").as_posix() for s in ("benign_eval", "attacks_static")]
    prompts = (P / "pools" / "prompts.parquet").as_posix()
    series = []
    for key, lab in CONFIGS:
        if key not in STYLE:
            continue
        ev = F.runs / f"{key}_{F.cfg['profile']}" / "events.parquet"
        if not ev.exists():
            continue
        t = np.sort(np.array([x for (x,) in duckdb.sql(f"""
            SELECT min(ts) FROM (
              SELECT g.campaign_id, unnest(string_split(e.response_hash, ',')) AS rh, e.ts
              FROM '{ev.as_posix()}' e JOIN read_parquet({gt}) g ON e.event_id = g.request_id
              JOIN '{prompts}' p USING (prompt_hash)
              WHERE g.population = 'attack' AND g.traffic_label = 'attack' AND e.blocked = 0 AND e.form = 'full'
                AND p.pool IN {CANARY})
            GROUP BY campaign_id, rh""").fetchall()]))
        series.append((key, lab, t / 86400))
    if not series:
        return "no events.parquet for none_plain / none_coupled / full_coupled"
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for key, lab, t in series:
        col, _, ls, _ = STYLE[key]
        ax.step(t, np.arange(1, len(t) + 1), where="post", color=col, ls=ls, label=f"{lab} ({len(t):,})")
    ax.set_xlim(0, F.cfg["scale"]["days"])
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Cumulative Q_eff")
    ax.yaxis.set_major_formatter(KFMT)
    ax.legend(loc="upper left")
    F.save(fig, "fig_yield_over_time", [f"runs/*_{F.cfg['profile']}/events.parquet", "traffic/*/ground_truth.parquet"])


def tuning(F):
    """Threshold tuning on benign_tune: benign shares across the p_tier2 grid (p_tier3 = p_tier2/10)."""
    f = F.cfg["processed"] / "reference" / f"policy_{F.cfg['versions']['detector']}_{F.cfg['profile']}.json"
    t = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if "sweep" not in t:
        return "tuning output without sweep (rerun experiments.tune)"
    p = np.array([r["p_tier2"] for r in t["sweep"]])
    P = F.cfg["policy"]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(p, [100 * r["share_tier2"] for r in t["sweep"]], color=S1, marker="o", ms=3, label="Reached tier 2")
    ax.plot(p, [100 * r["share_tier3"] for r in t["sweep"]], color=S2, marker="s", ms=3, ls="--", label="Reached tier 3")
    for cap, lab in [(P["cap_tier2"], "tier-2 cap"), (P["cap_tier3"], "tier-3 cap")]:
        ax.axhline(100 * cap * P["tune_margin"], color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"{lab} × {P['tune_margin']:g}", (p.min(), 100 * cap * P["tune_margin"]), xytext=(0, 2),
                    textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.axvline(t["thresholds"]["p_tier2"], color=INK, lw=1)
    ax.annotate(f"chosen p_tier2 = {t['thresholds']['p_tier2']:.0e}", (t["thresholds"]["p_tier2"], ax.get_ylim()[1]),
                xytext=(3, -10), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("p_tier2 threshold (stricter →)")
    ax.set_ylabel("Benign tuning accounts (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    F.save(fig, "fig_threshold_tuning", [str(f.relative_to(ROOT))],
           f"{t['accounts']} accounts of {t['stream']}; observe-mode replay, offline policy simulation")


def global_monitor(F):
    """Kesarwani-style global coverage: trailing-7-day distinct prompts per canary capability vs threshold."""
    f = F.runs / f"global_monitor_{F.cfg['profile']}.json"
    g = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if "series" not in g:
        return "global monitor output without series (rerun experiments.baselines global)"
    caps = [c for c in CANARY if c in g["series"]]
    fig, axes = plt.subplots(1, len(caps), figsize=(WIDE, 2.0), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), caps):
        d = np.arange(len(g["series"][c]["all"]))
        ax.plot(d, g["series"][c]["all"], color=S1, label="All traffic")
        ax.plot(d, g["series"][c]["benign"], color=S2, ls="--", label="Benign traffic only")
        ax.axhline(g["thresholds"][c], color=INK, lw=1, ls=":")
        ax.annotate("alarm threshold (benign ref. p99)", (d[-1], g["thresholds"][c]), xytext=(0, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color=INK)
        ax.set_title({"canary_kb": "Canary knowledge base", "canary_lang": "Canary language (Kesh)"}[c], loc="left")
        ax.set_xlabel("Simulated day")
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel("Distinct prompts, 7-day window")
    np.atleast_1d(axes)[-1].legend(loc="lower right")
    F.save(fig, "fig_global_monitor", [str(f.relative_to(ROOT))],
           f"false-alarm days on benign-only traffic: {g['benign_only_false_alarm_days']} of {g['days']}")


def search(F):
    """Best-response search: cost per trial (reached vs extrapolated) and running best, per defense."""
    runs = []
    for d in ("none", "full"):
        s = F.runs / f"search_{d}_{F.cfg['profile']}.json"
        if s.exists() and "bound" in (meta := json.loads(s.read_text(encoding="utf-8"))):
            rows = [json.loads(x) for x in (F.runs / f"search_{d}_{F.cfg['profile']}.jsonl").read_text().splitlines()]
            runs.append((d, meta, rows))
    if not runs:
        return "no search summary with the current (bounded) objective; run experiments.search"
    fig, axes = plt.subplots(1, len(runs), figsize=(COL * len(runs), 2.3), sharey=True, squeeze=False)
    for ax, (d, meta, rows) in zip(axes[0], runs):
        x = np.array([r["trial"] for r in rows])
        o = np.array([r["objective"] for r in rows])
        ok = np.array([r["reached"] for r in rows])
        ax.scatter(x[~ok], o[~ok], s=14, facecolors="none", edgecolors=MUTED, label="Short of target (extrapolated)")
        ax.scatter(x[ok], o[ok], s=16, color=S1, label="Reached target")
        best = np.minimum.accumulate(np.where(ok, o, np.inf))
        if np.isfinite(best).any():
            ax.step(x, best, where="post", color=S1, lw=1.2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(PLAIN)
        ax.set_title(f"{'No defense' if d == 'none' else 'Full gateway'}: target n = {meta['target_n']:,} "
                     f"({meta['bound']})", loc="left")
        ax.set_xlabel("Trial")
    axes[0][0].set_ylabel("Attacker cost to target")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    F.save(fig, "fig_best_response_search", [f"runs/search_*_{F.cfg['profile']}.json(l)"],
           json.dumps({d: m.get("dcm_best_response") for d, m, _ in runs}))


def latency(F):
    """Added hot-path latency (p50 / p95 / p99) per configuration against the 10-20 ms target."""
    reps = [(l, r) for l, r in F.reports() if r["hot_path_ms"]["n_live"]]
    if not reps:
        return "no live-encoded latency samples"
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    y = np.arange(len(reps))
    ax.axvspan(10, 20, color=GRID, lw=0, label="target 10–20 ms")
    for k, col, mk in zip(("p50", "p95", "p99"), RAMP, ("o", "s", "D")):
        ax.scatter([r["hot_path_ms"][k] for _, r in reps], y, color=col, marker=mk, s=20, label=k, zorder=3)
    ax.set_yticks(y, [l for l, _ in reps])
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Added latency per request (ms, CPU)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    F.save(fig, "fig_latency", [f"runs/*_{F.cfg['profile']}/report.json"],
           f"{reps[0][1]['hot_path_ms']['n_live']} live-encoded requests per run; teacher call excluded")


def dcm(F):
    """Defense cost multiplier per configuration at tau = 0.5 and 0.8, both bounds, range over curve seeds."""
    reps = [(l, r) for l, r in F.reports() if r.get("dcm") and not (r["defense"] == "none" and not r["coupled"])]
    if not reps:
        return "no DCM yet (needs the step-5 curve and a none_plain run)"
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2), sharey=True)
    y = np.arange(len(reps))
    labeled = set()
    for ax, t in zip(axes, ("0.5", "0.8")):
        for b, col, mk, off in zip(BOUNDS, (S1, S2), ("o", "s"), (-0.15, 0.15)):
            for i, (_, r) in enumerate(reps):
                v = [x for x in r["dcm"].get(t, {}).get(b, {}).values() if x is not None]
                if v:
                    ax.errorbar(np.median(v), i + off, xerr=[[np.median(v) - min(v)], [max(v) - np.median(v)]], fmt=mk,
                                color=col, ms=4.5, capsize=2, elinewidth=1, label=None if b in labeled else b)
                    labeled.add(b)
                else:
                    ax.annotate("not reached", (1, i + off), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                                color=MUTED, va="center")
        ax.axvline(1, color=MUTED, lw=0.8)
        ax.axvline(2, color=INK, lw=1, ls="--")
        ax.annotate("H3 bar (2×)", (2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PLAIN)
        ax.xaxis.set_minor_formatter(PLAIN)
        ax.set_ylim(len(reps) - 0.5, -0.5)
        ax.set_title(f"τ = {t}", loc="left", pad=12)
        ax.set_xlabel("DCM = C_τ(defended) / C_τ(none)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [l for l, _ in reps])
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h, l = h + [hh], l + [ll]
    fig.legend(h, [f"{x} bound" for x in l], loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    F.save(fig, "fig_dcm_static", [f"runs/*_{F.cfg['profile']}/report.json"],
           "static attackers; point = median over curve seeds, bar = min-max")


LAYERS = [("l0_plain", "D1 rules"), ("d2_plain", "D2 + account"), ("d3_plain", "D3 + cluster"),
          ("d4_plain", "D4 + tiers"), ("full_coupled", "D5 + coupled")]


def layers(F):
    """Stage 3 layer ablation: attacker yield (Q_eff, 95% CI over campaigns) and benign tier-2 share per layer."""
    reps = [(l, r) for k, l in LAYERS if (r := F.report(k))]
    if len(reps) < 2:
        return "stage 3 reports (d2 / d3 / d4) missing"
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 1.9), sharey=True)
    y = np.arange(len(reps))
    q = np.array([r["attacker"]["Q_eff_total"] for _, r in reps], float)
    ci = np.array([r["attacker"]["Q_eff_total_ci95"] for _, r in reps], float)
    axes[0].barh(y, q, height=0.6, color=S1)
    axes[0].errorbar(q, y, xerr=[q - ci[:, 0], ci[:, 1] - q], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    axes[0].set_xlabel("Unique canary responses (Q_eff), summed over campaigns")
    axes[0].xaxis.set_major_formatter(KFMT)
    v = [100 * r["benign_cost"]["share_tier2"] for _, r in reps]
    label_bars(axes[1], axes[1].barh(y, v, height=0.6, color=S2), lambda w: f"{w:.2f}%", clear=1.0)
    axes[1].axvline(1.0, color=INK, lw=1, ls="--")
    axes[1].set_xlim(0, max(max(v), 1.0) * 1.3)
    axes[1].set_xlabel("Benign accounts reaching tier 2 or above (%)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [l for l, _ in reps])
    axes[0].invert_yaxis()
    F.save(fig, "fig_layers", [f"runs/*_{F.cfg['profile']}/report.json"],
           "stage 3 (README §12.3 rule 6); D2/D3 block at p_tier3, D4 graded tiers; dashed line = 1% cap")


def validation(F):
    """Step 16: measured student fidelity against the curve prediction (lo and hi bound), plus stage 8 runs."""
    f = F.runs / f"validate_{F.cfg['profile']}.json"
    if not f.exists():
        return "validate summary missing (GPU step 16)"
    v = json.loads(f.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(COL, COL))
    ax.plot([0, 1.05], [0, 1.05], color=MUTED, lw=0.8)
    ax.fill_between([0, 1.05], [-0.1, 0.95], [0.1, 1.15], color=GRID, lw=0, label="±0.10")
    for b, c, m in (("lo", S1, "o"), ("hi", S2, "s")):
        ax.scatter([x["predicted"][b] for x in v["cells"]], [x["measured"] for x in v["cells"]], color=c, marker=m,
                   s=16, label=f"cells, {b} prediction", zorder=3)
    g = F.runs / f"stage8_{F.cfg['profile']}.json"
    if g.exists():
        s8 = json.loads(g.read_text(encoding="utf-8"))["seeds"]
        ax.scatter([x["predicted"]["hi"] for x in s8], [x["fidelity"] for x in s8], color=S3, marker="^", s=22,
                   label="stage 8 student-guided (hi)", zorder=4)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Curve-predicted fidelity")
    ax.set_ylabel("Measured student fidelity")
    ax.legend(loc="upper left")
    F.save(fig, "fig_validation", [str(f.relative_to(ROOT))],
           f"closer bound {v['closer_bound']}, {v['within_0.10']} of {len(v['cells'])} cells within 0.10, "
           f"gate {'passed' if v['gate_passed'] else 'not passed'}")


FIGURES = {
    "scaling": scaling, "benign": benign_cost, "yield": yield_bars, "layers": layers, "validation": validation,
    "yield_stage": lambda F: yield_bars(F, STAGE, "fig_attacker_yield_stage"),
    "detection": detection, "detection_stage": lambda F: detection(F, STAGE, "fig_detection_stage"),
    "campaigns": campaigns, "campaigns_stage": lambda F: campaigns(F, STAGE, "fig_campaign_yield_stage"),
    "yield_time": yield_time, "tuning": tuning, "global": global_monitor, "search": search,
    "latency": latency, "dcm": dcm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FIGURES))
    a = ap.parse_args()
    F = Figs(load())
    for name in a.only or FIGURES:
        why = FIGURES[name](F)
        if why:
            print(f"  skipped {name}: {why}")
    if F.manifest:
        m = F.out / "manifest.json"
        old = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        m.write_text(json.dumps({**old, **F.manifest}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
