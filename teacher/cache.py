"""API teacher: key check, step-2 gate, step-4 response cache. Runs on CPU (laptop or Colab); resumable.

  python -m teacher.cache check   one tiny call per key of every candidate provider
  python -m teacher.cache gate    canary accuracy of each candidate on held-out eval items -> cache/gate.json
  python -m teacher.cache build   pool prompts x (k_full full + k_answer_only answer-only) -> cache/teacher_cache.parquet

Selection rule and sizes: configs/default.toml [teacher_api]. The teacher only sees the records it is given
in-context (never items.parquet answers or private_notes). One model fills the whole cache.
"""
import datetime
import hashlib
import json
import sys

import duckdb

from configs import load
from teacher.backends import TOWN, ApiTeacher, QuotaExhausted
from teacher.canary import score



def by_town(rows, kb, col):
    """Sort so one batch holds one town's questions (fewest records in the system prompt)."""
    names = {e["name"] for e in kb}
    return sorted(rows, key=lambda r: (next((w for w in TOWN.findall(r[col]) if w in names), ""), r))


def setup():
    cfg = load()
    C = cfg["processed"] / "canary"
    kb = json.loads((C / "kb.json").read_text(encoding="utf-8"))
    lex = json.loads((C / "lexicon.json").read_text(encoding="utf-8"))
    out = cfg["processed"] / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return cfg, cfg["teacher_api"], (C / "items.parquet").as_posix(), kb, lex, out


def chunks(rows, n):
    return [rows[i:i + n] for i in range(0, len(rows), n)]


def check():
    cfg, T, _, kb, lex, _ = setup()
    for cand in T["candidates"]:
        prov, model = cand.split(":", 1)
        t = ApiTeacher(prov, model, kb, lex, max_wait_s=0, extra=T["extra"].get(cand))
        for i, key in enumerate(list(t.keys)):
            t.keys, t.ready = [key], [0.0]
            try:
                a = t.generate_batch(["Translate into Kesh: the red fox sees a small bird"], 0, "answer_only")[0]
                print(f"OK   {cand} key#{i + 1}: {a!r:.60}")
            except Exception as e:
                print(f"FAIL {cand} key#{i + 1}: {str(e)[:120]}")


def gate():
    cfg, T, items, kb, lex, out = setup()
    f = out / "gate.json"
    prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    res = prev.get("results", {}) if prev.get("template") == T["template"] else {}  # new prompt -> regate all
    batch = {"canary_kb": T["batch_kb"], "canary_lang": T["batch_lang"]}
    for cand in T["candidates"]:
        if cand in res and "accuracy" in res[cand]:
            continue
        prov, model = cand.split(":", 1)
        try:
            t, acc, missing = ApiTeacher(prov, model, kb, lex, max_wait_s=120, extra=T["extra"].get(cand)), {}, 0
            for cap in batch:
                rows = by_town(duckdb.sql(f"""SELECT prompt, answer FROM '{items}' WHERE split = 'eval'
                    AND capability = '{cap}' ORDER BY md5(item_id) LIMIT {T['gate_n']}""").fetchall(), kb, 0)
                hits = []
                for ch in chunks(rows, batch[cap]):
                    ans = t.generate_batch([p for p, _ in ch], 0, "answer_only", T["temperature"])
                    missing += sum(a is None for a in ans)
                    hits += [score(a or "", gold) for a, (_, gold) in zip(ans, ch)]  # no answer counts as wrong
                acc[cap] = round(sum(hits) / len(hits), 3)
            res[cand] = {"accuracy": acc, "missing": missing, "version": t.version}
        except (QuotaExhausted, RuntimeError) as e:
            res[cand] = {"error": str(e)[:300]}
        print(cand, res[cand], flush=True)
        f.write_text(json.dumps({"results": res, "template": T["template"]}, indent=1), encoding="utf-8")
    ok = [c for c in T["candidates"] if "accuracy" in res.get(c, {}) and min(res[c]["accuracy"].values()) >= T["gate"]
          and (T["allow_closed"] or c not in T["closed"])]
    teacher = max(ok, key=lambda c: (min(res[c]["accuracy"].values()), -T["candidates"].index(c)), default=None)
    rule = f"max min(kb, lang) accuracy among candidates with both >= {T['gate']}; closed allowed: {T['allow_closed']}"
    f.write_text(json.dumps({"results": res, "teacher": teacher, "rule": rule, "gate_n": T["gate_n"],
                             "template": T["template"], "date": str(datetime.date.today())}, indent=1), encoding="utf-8")
    print("TEACHER:", teacher or "none passed the gate; no cache will be built")
    return teacher


def build():
    cfg, T, items, kb, lex, out = setup()
    g = out / "gate.json"
    teacher = json.loads(g.read_text(encoding="utf-8")).get("teacher") if g.exists() else None
    if not teacher:
        sys.exit("no gated teacher: run `python -m teacher.cache gate` first")
    prov, model = teacher.split(":", 1)
    t = ApiTeacher(prov, model, kb, lex, max_wait_s=900, extra=T["extra"].get(teacher))
    jl = out / f"responses_{hashlib.sha1((t.version + T['template']).encode()).hexdigest()[:8]}.jsonl"
    done = set()
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["item_id"], r["form"], r["k"]))
    pool = {cap: by_town(duckdb.sql(f"SELECT item_id, capability, prompt, answer FROM '{items}' WHERE split = 'pool' "
                                    f"AND capability = '{cap}'").fetchall(), kb, 2)
            for cap in ("canary_kb", "canary_lang")}
    batch = {"canary_kb": T["batch_kb"], "canary_lang": T["batch_lang"]}
    jobs = [(form, k, cap, ch) for form, K in (("full", T["k_full"]), ("answer_only", T["k_answer_only"]))
            for k in range(K) for cap in pool for ch in chunks(pool[cap], batch[cap])]
    today, complete = str(datetime.date.today()), True
    with jl.open("a", encoding="utf-8") as fh:
        try:
            for attempt in range(3):  # later passes retry items the model skipped in its JSON reply
                todo = [(form, k, cap, [r for r in ch if (r[0], form, k) not in done]) for form, k, cap, ch in jobs]
                todo = [j for j in todo if j[3]]
                print(f"pass {attempt + 1}: {len(todo)} calls to go", flush=True)
                for n, (form, k, cap, ch) in enumerate(todo):
                    seed = int(hashlib.sha1(f"{form}|{k}|{ch[0][0]}".encode()).hexdigest()[:8], 16)
                    for (iid, c, prompt, _), a in zip(ch, t.generate_batch([r[2] for r in ch], seed, form, T["temperature"])):
                        if a is None:
                            continue
                        fh.write(json.dumps({"item_id": iid, "capability": c, "prompt": prompt, "form": form, "k": k,
                                             "response": a, "seed": seed, "temperature": T["temperature"],
                                             "teacher": t.version, "template": T["template"], "created": today}) + "\n")
                        done.add((iid, form, k))
                    fh.flush()
                    if n % 50 == 0:
                        print(f"  {n}/{len(todo)} calls, {len(done)} responses", flush=True)
        except QuotaExhausted as e:
            print(f"STOPPED (resume later, nothing is lost): {e}")
            complete = False
    need = sum(len(v) for v in pool.values()) * (T["k_full"] + T["k_answer_only"])
    complete = complete and len(done) >= need
    pq = (out / "teacher_cache.parquet").as_posix()
    duckdb.sql(f"COPY (SELECT * FROM read_json_auto('{jl.as_posix()}')) TO '{pq}' (FORMAT parquet)")
    hit = duckdb.sql(f"""SELECT c.form, avg(CASE WHEN contains(lower(c.response), lower(i.answer)) THEN 1 ELSE 0 END)
        FROM '{pq}' c JOIN '{items}' i USING (item_id) GROUP BY ALL ORDER BY ALL""").fetchall()
    meta = {"teacher": t.version, "template": T["template"], "responses": len(done), "expected": need,
            "complete": complete, "answer_contained": {f: round(v, 3) for f, v in hit}, "k_full": T["k_full"],
            "k_answer_only": T["k_answer_only"], "date": today}
    (out / "teacher_cache.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    {"check": check, "gate": gate, "build": build}[sys.argv[1]]()
