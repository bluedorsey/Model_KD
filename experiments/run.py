"""Replay one defense configuration over benign_eval + attack traffic and report README §12.3 metrics.

  .venv/Scripts/python.exe -m experiments.run --defense none      # stage 1: no defense
  .venv/Scripts/python.exe -m experiments.run --defense l0        # stage 1: quotas only
  .venv/Scripts/python.exe -m experiments.run --defense mmd       # stage 2: per-account MMD only
  .venv/Scripts/python.exe -m experiments.run --defense seat      # stage 2: SEAT-style similar-pair count
  .venv/Scripts/python.exe -m experiments.run --defense simlink   # stage 2: cross-account similarity linkage
  .venv/Scripts/python.exe -m experiments.run --defense full      # rules + account + cluster + tiers + coupled
  .venv/Scripts/python.exe -m experiments.run --defense d2        # stage 3 layers d1..d5 (README §12.3 rule 6)
  .venv/Scripts/python.exe -m experiments.run --defense full --off infra --streams benign_eval attacks_distributed
  .venv/Scripts/python.exe -m experiments.run --defense full --lr  # stage 6: + LR (experiments/fit_lr.py)

Q_eff counts unique full-quality canary responses. Fidelity maps the attacker's training set to the step-5
data-scaling curve (runs/scaling/results.jsonl), whose training sets hold distinct prompts first. The
attacker's set is mapped with two bounds: lo = distinct canary prompts (extra samples of a prompt teach
nothing), hi = Q_eff (every unique response is worth a new prompt). Step 16 checks which is closer. Without
the curve, fidelity and DCM are null. DCM_tau = C_tau(this defense) / C_tau(none_plain), each the cheapest
static campaign reaching tau x teacher accuracy; one value per bound and curve seed. Tuned tier thresholds
(experiments/tune.py) are applied when present.
"""
import argparse
import json
import math
import pickle

import duckdb
import numpy as np

from configs import ROOT, load
from gateway.replay import World, ref_path, run
from gateway.schema import DAY, EVENT, write

CANARY = ("canary_kb", "canary_lang")
TAUS = (0.5, 0.8)
BOUNDS = ("lo", "hi")
DEFENSES = ["none", "l0", "mmd", "seat", "simlink", "full", "d1", "d2", "d3", "d4", "d5"]
# stage 3 layers -> (replay mode, block-only policy, coupled sampling)
LAYERS = {"d1": ("l0", False, False), "d2": ("account", True, False), "d3": ("full", True, False),
          "d4": ("full", False, False), "d5": ("full", False, True)}
STAGE6_TRAIN = ("A01", "A02", "A04", "A16")  # LR training families; every other family is unseen


def configure(cfg, defense, coupled="auto"):
    """Set block-only policy and coupled sampling for `defense`; -> replay mode. coupled: auto | on | off
    (auto: on for full and d5 only; 'none --coupled on' is the stage-2 coupled-sampling-alone baseline)."""
    mode, block, coup = LAYERS.get(defense, (defense, False, defense == "full"))
    cfg["policy"]["block_only"] = block
    cfg["gateway"]["coupled"] = coupled == "on" or (coupled == "auto" and coup)
    return mode


def tpr_at_fpr(sb, sa, fpr, rng, boots=500):
    """Account-level TPR at a benign-account FPR, with a bootstrap 95% CI over accounts."""
    def one(b, a):
        t = np.quantile(b, 1 - fpr, method="higher")
        return float((a > t).mean()) if len(a) else float("nan")
    est = one(sb, sa)
    bs = [one(rng.choice(sb, len(sb)), rng.choice(sa, len(sa))) for _ in range(boots)]
    return {"tpr": round(est, 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]}


def auprc(sb, sa, prevalence):
    """Average precision with benign/attack accounts reweighted to the stated attacker prevalence."""
    s = np.concatenate([sa, sb])
    w = np.concatenate([np.full(len(sa), prevalence / len(sa)), np.full(len(sb), (1 - prevalence) / len(sb))])
    y = np.concatenate([np.ones(len(sa)), np.zeros(len(sb))])
    o = np.argsort(-s, kind="stable")
    tp, fp = np.cumsum(w[o] * y[o]), np.cumsum(w[o] * (1 - y[o]))
    prec = tp / np.maximum(tp + fp, 1e-300)
    return round(float((prec * w[o] * y[o]).sum() / prevalence), 4)


def curve(cfg):
    """Step-5 curve -> {"model", "teacher_acc", "seeds": {seed: (ns, accs)}} or None before it is measured.
    Accuracy = mean over the two canary capabilities; (n=1, acc=0) anchors the untrained student."""
    f, g = ROOT / "runs" / "scaling" / "results.jsonl", cfg["processed"] / "cache" / "gate.json"
    if not (f.exists() and g.exists()):
        return None
    model = cfg["student"]["curve_model"]
    rows = [r for r in map(json.loads, f.read_text().splitlines()) if r["model"] == model]
    gate = json.loads(g.read_text(encoding="utf-8"))
    if not rows or not gate.get("teacher"):
        return None
    mean = lambda acc: float(np.mean(list(acc.values()))) if isinstance(acc, dict) else float(acc)
    seeds = {}
    for r in sorted(rows, key=lambda r: r["n"]):
        seeds.setdefault(r["seed"], ([1], [0.0]))
        seeds[r["seed"]][0].append(r["n"])
        seeds[r["seed"]][1].append(mean(r["accuracy"]))
    seeds = {s: (np.array(n, float), np.maximum.accumulate(a)) for s, (n, a) in seeds.items()}
    return {"model": model, "teacher": gate["teacher"], "teacher_acc": mean(gate["results"][gate["teacher"]]["accuracy"]),
            "seeds": seeds}


def acc_at(ns, accs, n):
    return float(np.interp(np.log(max(n, 1)), np.log(ns), accs))


def n_for(ns, accs, target):
    """Smallest n (linear in log n between measured sizes) with accuracy >= target; None if never reached."""
    if accs[-1] < target:
        return None
    i = int(np.argmax(accs >= target))
    if i == 0:
        return int(ns[0])
    x0, x1, y0, y1 = math.log(ns[i - 1]), math.log(ns[i]), accs[i - 1], accs[i]
    return int(math.ceil(math.exp(x0 + (target - y0) / (y1 - y0) * (x1 - x0))))


def tau_targets(cv):
    """{(tau, seed): n_tau or None}."""
    return {(t, s): n_for(ns, a, t * cv["teacher_acc"]) for t in TAUS for s, (ns, a) in cv["seeds"].items()} if cv else {}


def campaign_yield(events, gt, pool_of, price, accounts, targets=()):
    """Per campaign: requests, unique responses (resp: first-seen times), distinct canary prompts, and the
    cost at which each bound first reached n, for (bound, n) in `targets`: lo = distinct prompts, hi = Q_eff.
    gt: event_id -> (label, attack, campaign, population). accounts: campaign -> accounts it pays for (all
    charged up front)."""
    out = {}
    for ev in events:
        label, attack, cid, pop = gt[ev["event_id"]]
        if pop != "attack":
            continue
        c = out.setdefault(cid, {"attack": attack, "requests": 0, "accounts": set(), "resp": {}, "prompts": set(),
                                 "cost_to": {}})
        c["requests"] += 1
        c["accounts"].add(ev["account_id"])
        if label == "attack" and not ev["blocked"] and ev["form"] == "full" and pool_of[ev["prompt_hash"]] in CANARY:
            for rh in ev["response_hash"].split(","):
                c["resp"].setdefault(rh, ev["ts"])
            c["prompts"].add(ev["prompt_hash"])
            have = {"lo": len(c["prompts"]), "hi": len(c["resp"])}
            for b, n in targets:
                if (b, n) not in c["cost_to"] and have[b] >= n:
                    c["cost_to"][b, n] = c["requests"] * price["query"] + accounts[cid] * price["account"]
    for c in out.values():
        c["n"] = {"lo": len(c["prompts"]), "hi": len(c["resp"])}
        c["cost"] = c["requests"] * price["query"] + len(c["accounts"]) * price["account"]
    return out


def read_gt(cfg, streams):
    T = cfg["processed"] / "traffic"
    files = ", ".join(f"'{(T / n / 'ground_truth.parquet').as_posix()}'" for n in streams)
    return {r[0]: r[1:] for r in duckdb.sql(f"SELECT request_id, traffic_label, attack_type, campaign_id, population "
                                            f"FROM read_parquet([{files}])").fetchall()}


def report(cfg, world, events, summary, gw, defense, streams):
    T = cfg["processed"] / "traffic"
    gt = read_gt(cfg, streams)
    camp_meta = {c["campaign_id"]: c for n in streams if n.startswith("attacks_")
                 for c in json.loads((T / n / "campaigns.json").read_text(encoding="utf-8"))["campaigns"]}
    cv = curve(cfg)
    targets = tau_targets(cv)
    camps = campaign_yield(events, gt, world.pool_of, cfg["cost"], {c: m["accounts"] for c, m in camp_meta.items()},
                           {(b, n) for n in targets.values() if n for b in BOUNDS})
    pop_of = {ev["account_id"]: gt[ev["event_id"]][3] for ev in events}
    family = {ev["account_id"]: gt[ev["event_id"]][1] for ev in events if gt[ev["event_id"]][3] == "attack"}
    camp_rows = {}
    for cid, c in sorted(camps.items()):
        first = min([summary[a]["first_t2"] for a in c["accounts"] if a in summary and summary[a]["first_t2"]],
                    default=None)
        q = len(c["resp"])
        fid = ({b: round(float(np.mean([acc_at(ns, a, c["n"][b]) for ns, a in cv["seeds"].values()]))
                         / cv["teacher_acc"], 4) for b in BOUNDS} if cv else None)
        camp_rows[cid] = {"attack": c["attack"], "accounts": len(c["accounts"]), "requests": c["requests"],
                          "Q_eff": q, "distinct_canary_prompts": len(c["prompts"]),
                          "fidelity": fid, "cost": round(c["cost"], 2),
                          "yield_before_enforcement": round(sum(t < first for t in c["resp"].values()) / q, 4)
                          if (q and first is not None) else (1.0 if q else None),
                          "enforced": first is not None, "planned_requests": camp_meta[cid]["requests"]}
    fidelity = None
    if cv:
        fidelity = {"curve_model": cv["model"], "teacher": cv["teacher"], "teacher_acc": round(cv["teacher_acc"], 4),
                    "tau": {}}
        for (t, s), n in sorted(targets.items()):
            for b in BOUNDS:
                costs = {cid: c["cost_to"][b, n] for cid, c in camps.items() if n and (b, n) in c["cost_to"]}
                best = min(costs, key=costs.get) if costs else None
                fidelity["tau"].setdefault(str(t), {}).setdefault(b, {})[str(s)] = {
                    "n_tau": n, "C_tau": round(costs[best], 2) if best else None, "campaign": best}
    benign = [a for a, p in pop_of.items() if p != "attack"]
    attack = [a for a, p in pop_of.items() if p == "attack"]
    get = lambda a, k, d: summary.get(a, {}).get(k, d)
    by_pop = {}
    for a in benign:
        b = by_pop.setdefault(pop_of[a], [0, 0, 0, 0])
        t = get(a, "max_tier", 0)
        b[0] += 1; b[1] += t >= 2; b[2] += t >= 3; b[3] += get(a, "verifications", 0)
    first, rec = {}, {}  # benign: first request served at tier >= 2, then days until one is served at tier <= 1
    for ev in events:
        a = ev["account_id"]
        if pop_of[a] == "attack" or a in rec:
            continue
        if a not in first:
            if ev["tier"] >= 2:
                first[a] = ev["ts"]
        elif ev["tier"] <= 1:
            rec[a] = (ev["ts"] - first[a]) / DAY
    for a in first:
        by_pop[pop_of[a]].append(rec.get(a))
    n_b = len(benign)
    rng = np.random.default_rng(0)
    score = lambda accts, key: np.array([-np.log10(get(a, key, 1.0)) for a in accts])
    det = None
    if attack:
        det = {}
        for name, key in [("account", "min_p"), ("cluster", "min_pc")]:
            sb, sa = score(benign, key), score(attack, key)
            det[name] = {"tpr@1%fpr": tpr_at_fpr(sb, sa, 0.01, rng), "tpr@0.1%fpr": tpr_at_fpr(sb, sa, 0.001, rng),
                         "auprc@1%": auprc(sb, sa, 0.01), "auprc@0.1%": auprc(sb, sa, 0.001)}
        groups = {f: [a for a in attack if family[a] == f] for f in sorted(set(family.values()))}
        groups["unseen"] = [a for a in attack if family[a] not in STAGE6_TRAIN]
        det["by_attack"] = {f: {name: tpr_at_fpr(score(benign, key), score(accts, key), 0.01, rng)
                                for name, key in [("account", "min_p"), ("cluster", "min_pc")]}
                            for f, accts in groups.items() if accts}
    lat = np.array(gw.latency_ms) if gw.latency_ms else np.array([np.nan])
    q_by = np.array([r["Q_eff"] for r in camp_rows.values()], float)
    bq = [rng.choice(q_by, len(q_by)).sum() for _ in range(1000)] if len(q_by) else [0]
    return {
        "defense": defense, "profile": cfg["profile"], "coupled": cfg["gateway"]["coupled"], "streams": streams,
        "versions": {"model": gw.versions[0], "encoder": gw.versions[1], "detector": gw.versions[2],
                     "policy": gw.versions[3], "pools": cfg["versions"]["pools"]},
        "policy_thresholds": {k: cfg["policy"][k] for k in ("p_tier1", "p_tier2", "p_tier3")},
        "variant": {"block_only": cfg["policy"].get("block_only", False), "lr": cfg["scoring"].get("use_lr", False),
                    "linkage_off": cfg["linkage"].get("off", [])},
        "events": len(events),
        "attacker": {"Q_eff_total": int(q_by.sum()),
                     "Q_eff_total_ci95": [round(float(np.quantile(bq, 0.025))), round(float(np.quantile(bq, 0.975)))],
                     "campaigns": camp_rows},
        "fidelity": fidelity,
        "benign_cost": {"accounts": n_b,
                        "share_tier2": round(sum(v[1] for v in by_pop.values()) / max(n_b, 1), 5),
                        "share_tier3": round(sum(v[2] for v in by_pop.values()) / max(n_b, 1), 5),
                        "by_population": {p: {"accounts": v[0], "tier2": v[1], "tier3": v[2], "verifications": v[3],
                                              "recovery_days_median": recovery(v[4:]),
                                              "never_recovered": sum(x is None for x in v[4:])}
                                          for p, v in sorted(by_pop.items())}},
        "detection": {"benign_accounts": n_b, "attack_accounts": len(attack), **det} if det else None,
        "hot_path_ms": {"n_live": int(len(gw.latency_ms)), "p50": round(float(np.percentile(lat, 50)), 2),
                        "p95": round(float(np.percentile(lat, 95)), 2), "p99": round(float(np.percentile(lat, 99)), 2)},
    }


def recovery(days):
    """Median days to recover over accounts that recovered (None = never)."""
    got = [x for x in days if x is not None]
    return round(float(np.median(got)), 3) if got else None


def dcm(rep, base):
    """{tau: {bound: {seed: C_tau(rep) / C_tau(base)}}}; None where either side never reaches tau (censored)."""
    if not (rep.get("fidelity") and base.get("fidelity")):
        return None
    out = {}
    for t, by_bound in rep["fidelity"]["tau"].items():
        for bd, by_seed in by_bound.items():
            for s, r in by_seed.items():
                b = base["fidelity"]["tau"].get(t, {}).get(bd, {}).get(s, {})
                out.setdefault(t, {}).setdefault(bd, {})[s] = (round(r["C_tau"] / b["C_tau"], 3)
                                                              if r["C_tau"] and b.get("C_tau") else None)
    return out


def policy_path(cfg):
    lr = "+lr" if cfg["scoring"].get("use_lr") else ""
    return cfg["processed"] / "reference" / f"policy_{cfg['versions']['detector']}{lr}_{cfg['profile']}.json"


def tuned(cfg):
    """Apply experiments/tune.py thresholds for this detector version (+lr) and profile, if they exist."""
    f = policy_path(cfg)
    if f.exists():
        t = json.loads(f.read_text(encoding="utf-8"))
        cfg["policy"].update(t["thresholds"])
        cfg["versions"]["policy"] += f"+tuned-{t['stream']}"
    return cfg


def setup(defense, coupled="auto", lr=False, off=(), untuned=False):
    """-> (cfg, replay mode): tuned thresholds (with the LR if lr), linkage switches, layer and coupled sampling."""
    cfg = load()
    cfg["scoring"]["use_lr"] = lr
    cfg["linkage"]["off"] = list(off)
    cfg = cfg if untuned else tuned(cfg)
    return cfg, configure(cfg, defense, coupled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", default="full", choices=DEFENSES)
    ap.add_argument("--streams", nargs="+", default=["benign_eval", "attacks_static"])
    ap.add_argument("--coupled", default="auto", choices=["auto", "on", "off"],
                    help="auto: on for full and d5 only; 'none --coupled on' is the stage-2 coupled-sampling-alone baseline")
    ap.add_argument("--untuned", action="store_true", help="ignore tuned thresholds (config defaults)")
    ap.add_argument("--off", nargs="+", default=[], choices=["infra", "timing", "semantic", "sharding"],
                    help="stage 4: linkage signal family switched off")
    ap.add_argument("--lr", action="store_true", help="stage 6: account scorer + LR, thresholds tuned with it")
    ap.add_argument("--live-every", type=int, default=500)
    a = ap.parse_args()
    cfg, mode = setup(a.defense, a.coupled, a.lr, a.off, a.untuned)
    world = World(cfg)
    ref = pickle.loads(ref_path(cfg).read_bytes())
    events, summary, gw, _ = run(cfg, world, a.streams, mode, ref, live_every=a.live_every)
    tag = "" if a.streams == ["benign_eval", "attacks_static"] else "_" + "+".join(a.streams)
    var = "".join(f"_off-{x}" for x in a.off) + ("_lr" if a.lr else "")
    name = f"{a.defense}_{'coupled' if cfg['gateway']['coupled'] else 'plain'}{var}_{cfg['profile']}{tag}"
    out = cfg["processed"] / "runs" / name
    write([[e[c] for c in EVENT] for e in events], EVENT, out / "events.parquet")
    rep = report(cfg, world, events, summary, gw, a.defense, a.streams)
    base = cfg["processed"] / "runs" / f"none_plain_{cfg['profile']}{tag}" / "report.json"
    rep["dcm"] = dcm(rep, json.loads(base.read_text(encoding="utf-8"))) if base.exists() else None
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("defense", "events", "policy_thresholds", "benign_cost", "detection",
                                          "hot_path_ms", "fidelity", "dcm")}, indent=1))
    print(json.dumps({c: (r["attack"], r["accounts"], r["requests"], r["Q_eff"], r["enforced"])
                      for c, r in rep["attacker"]["campaigns"].items()}))


if __name__ == "__main__":
    main()
