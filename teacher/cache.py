"""Teacher gate (step 2) and response cache (step 4), local GPU teacher or API teacher; resumable.

  python -m teacher.cache check   one tiny call per key of every API candidate
  python -m teacher.cache gate    canary accuracy of each candidate on held-out eval items -> cache/gate.json
  python -m teacher.cache build   pool prompts x (k_full full + k_answer_only answer-only) -> cache/teacher_cache.parquet
  python -m teacher.cache verify  coverage, no canary secret, seed reproducibility (local) -> cache/verify.json

Candidates and sizes: configs/default.toml [teacher_api]; candidate "local" = [teacher_local] (Qwen2.5-7B 4-bit +
canary QLoRA adapter, README §6 primary). An API teacher only sees the records it is given in-context (never
items.parquet answers or private_notes). One model fills the whole cache.
"""
import datetime
import hashlib
import json
import sys
import time

import duckdb

from configs import ROOT, load
from teacher.backends import TOWN, ApiTeacher, LocalTeacher, QuotaExhausted, local_version
from teacher.canary import score

LOCAL_TEMPLATE = "local-adapter"  # the local teacher gets no in-context records: the adapter holds the canary


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


def adapter_path(cfg):
    return str(ROOT / cfg["teacher_local"]["adapter"]) if cfg["teacher_local"]["adapter"] else None


def local_expected(cfg):
    L = cfg["teacher_local"]
    return local_version(L["model"], adapter_path(cfg), "4bit-dq" if L["load_4bit"] else "fp16")


def make_teacher(cand, cfg, kb, lex, max_wait_s):
    if cand == "local":
        L = cfg["teacher_local"]
        local_expected(cfg)  # RuntimeError before any model download if the adapter is not trained yet
        return LocalTeacher(L["model"], adapter_path(cfg), L["load_4bit"])
    prov, model = cand.split(":", 1)
    return ApiTeacher(prov, model, kb, lex, max_wait_s=max_wait_s, extra=cfg["teacher_api"]["extra"].get(cand))


def read_done(jl):
    """Rows already written; a line cut short by a crash or kill is skipped (and regenerated)."""
    done = set()
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            done.add((r["item_id"], r["form"], r["k"]))
    return done


def row_seed(form, k, item_id):
    """Per-row seed: candidate k of a prompt is the same text whichever batch it was generated in."""
    return int(hashlib.sha1(f"{form}|{k}|{item_id}".encode()).hexdigest()[:15], 16)


def check():
    cfg, T, _, kb, lex, _ = setup()
    for cand in T["candidates"]:
        if cand == "local":
            continue
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
    for cand in T["candidates"]:
        if cand in res and "accuracy" in res[cand]:
            if cand != "local":
                continue
            try:  # local: regate only when the adapter (or base model) changed
                if res[cand]["version"] == local_expected(cfg):
                    continue
            except RuntimeError:
                pass
        n = cfg["teacher_local"]["batch"] if cand == "local" else None
        batch = {"canary_kb": n or T["batch_kb"], "canary_lang": n or T["batch_lang"]}
        try:
            t, acc, missing = make_teacher(cand, cfg, kb, lex, 120), {}, 0
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
            del t  # free the GPU before the next candidate
        except (QuotaExhausted, RuntimeError, ImportError, OSError) as e:  # no GPU / adapter / key / download
            res[cand] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(cand, res[cand], flush=True)
        f.write_text(json.dumps({"results": res, "template": T["template"]}, indent=1), encoding="utf-8")
    ok = [c for c in T["candidates"] if "accuracy" in res.get(c, {}) and min(res[c]["accuracy"].values()) >= T["gate"]
          and (T["allow_closed"] or c not in T["closed"])]
    teacher = "local" if "local" in ok else \
        max(ok, key=lambda c: (min(res[c]["accuracy"].values()), -T["candidates"].index(c)), default=None)
    rule = (f"local adapter teacher if both accuracies >= {T['gate']} (README §6 primary backend); else max "
            f"min(kb, lang) accuracy among API candidates with both >= {T['gate']}; closed allowed: {T['allow_closed']}")
    f.write_text(json.dumps({"results": res, "teacher": teacher, "rule": rule, "gate_n": T["gate_n"],
                             "template": T["template"], "date": str(datetime.date.today())}, indent=1), encoding="utf-8")
    print("TEACHER:", teacher or "none passed the gate; no cache will be built")
    return teacher is not None


def build():
    cfg, T, items, kb, lex, out = setup()
    g = out / "gate.json"
    gated = json.loads(g.read_text(encoding="utf-8")) if g.exists() else {}
    teacher = gated.get("teacher")
    if not teacher:
        sys.exit("no gated teacher: run `python -m teacher.cache gate` first")
    local = teacher == "local"
    t = make_teacher(teacher, cfg, kb, lex, 900)
    if local and t.version != gated["results"]["local"]["version"]:
        sys.exit(f"loaded teacher {t.version} is not the gated {gated['results']['local']['version']}: rerun gate")
    tpl = LOCAL_TEMPLATE if local else T["template"]
    jl = out / f"responses_{hashlib.sha1((t.version + tpl).encode()).hexdigest()[:8]}.jsonl"
    done = read_done(jl)
    pool = {cap: by_town(duckdb.sql(f"SELECT item_id, capability, prompt, answer FROM '{items}' WHERE split = 'pool' "
                                    f"AND capability = '{cap}'").fetchall(), kb, 2)
            for cap in ("canary_kb", "canary_lang")}
    today, complete, extra = str(datetime.date.today()), True, {}
    with jl.open("a", encoding="utf-8") as fh:
        if jl.stat().st_size and not jl.read_bytes().endswith(b"\n"):
            fh.write("\n")  # a torn last line from a kill would otherwise swallow the next row
        if local:
            extra = fill_local(t, cfg, pool, done, fh, today)
        else:
            complete = fill_api(t, T, pool, done, fh, today)
    need = sum(len(v) for v in pool.values()) * (T["k_full"] + T["k_answer_only"])
    pq = (out / "teacher_cache.parquet").as_posix()
    duckdb.sql(f"""COPY (SELECT * FROM read_json_auto('{jl.as_posix()}', ignore_errors = true)
        WHERE item_id IS NOT NULL AND response IS NOT NULL QUALIFY row_number() OVER (PARTITION BY item_id, form, k ORDER BY created) = 1) TO '{pq}' (FORMAT parquet)""")
    have = duckdb.sql(f"SELECT count(*) FROM '{pq}'").fetchone()[0]  # what is on disk, not what this run thinks
    complete = complete and have >= need
    hit = duckdb.sql(f"""SELECT c.form, avg(CASE WHEN contains(lower(c.response), lower(i.answer)) THEN 1 ELSE 0 END)
        FROM '{pq}' c JOIN '{items}' i USING (item_id) GROUP BY ALL ORDER BY ALL""").fetchall()
    if local:
        extra["truncated"], extra["empty"] = duckdb.sql(
            f"SELECT count(*) FILTER (WHERE truncated), count(*) FILTER (WHERE trim(response) = '') FROM '{pq}'").fetchone()
    meta = {"teacher": t.version, "template": tpl, "responses": have, "expected": need,
            "complete": complete, "answer_contained": {f: round(v, 3) for f, v in hit}, "k_full": T["k_full"],
            "k_answer_only": T["k_answer_only"], **extra, "date": today}
    (out / "teacher_cache.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    return complete


def fill_api(t, T, pool, done, fh, today):
    batch = {"canary_kb": T["batch_kb"], "canary_lang": T["batch_lang"]}
    jobs = [(form, k, cap, ch) for form, K in (("full", T["k_full"]), ("answer_only", T["k_answer_only"]))
            for k in range(K) for cap in pool for ch in chunks(pool[cap], batch[cap])]
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
        return False
    return True


def fill_local(t, cfg, pool, done, fh, today):
    """Batched Gumbel-max generation on the GPU. One form per batch, prompts sorted by length (less padding)."""
    import torch
    T, L = cfg["teacher_api"], cfg["teacher_local"]
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0, toks = time.time(), 0
    for form, K in (("full", T["k_full"]), ("answer_only", T["k_answer_only"])):
        todo = sorted(((k, r) for k in range(K) for cap in pool for r in pool[cap] if (r[0], form, k) not in done),
                      key=lambda x: (len(x[1][2]), x[0], x[1][0]))
        print(f"{form}: {len(todo)} generations to go", flush=True)
        for i, ch in enumerate(chunks(todo, L["batch"])):
            seeds = [row_seed(form, k, r[0]) for k, r in ch]
            outs = t.generate_rows([r[2] for _, r in ch], seeds, form, L[f"max_tokens_{form}"], T["temperature"])
            for (k, (iid, c, prompt, _)), seed, (text, trunc) in zip(ch, seeds, outs):
                fh.write(json.dumps({"item_id": iid, "capability": c, "prompt": prompt, "form": form, "k": k,
                                     "response": text, "seed": seed, "temperature": T["temperature"], "truncated": trunc,
                                     "teacher": t.version, "template": LOCAL_TEMPLATE, "created": today}) + "\n")
                done.add((iid, form, k))
                toks += len(t.tok(text, add_special_tokens=False)["input_ids"])
            fh.flush()
            if i % 20 == 0:
                print(f"  {min((i + 1) * L['batch'], len(todo))}/{len(todo)}, "
                      f"{toks / max(time.time() - t0, 1e-9):.0f} tok/s", flush=True)
    dt = time.time() - t0
    return {"tok_per_s": round(toks / dt, 1) if dt > 1 and toks else None, "batch": L["batch"],
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None,
            "gpu": torch.cuda.get_device_name(0) if cuda else "cpu"}


def verify():
    """Step-4 gate: every pool prompt has all forms, no canary secret or private field, and (local teacher) cached
    candidates regenerate from their seed one at a time, as the gateway would serve them."""
    cfg, T, items, kb, lex, out = setup()
    pq = (out / "teacher_cache.parquet").as_posix()
    mf = out / "teacher_cache.json"
    if not mf.exists():
        sys.exit("no cache: run `python -m teacher.cache build` first")
    meta = json.loads(mf.read_text(encoding="utf-8"))
    cols = [c[0] for c in duckdb.sql(f"DESCRIBE SELECT * FROM '{pq}'").fetchall()]
    missing = duckdb.sql(f"""SELECT count(*) FROM (SELECT i.item_id, f.form, f.k FROM '{items}' i,
        (SELECT 'full' form, range k FROM range({T['k_full']})
         UNION ALL SELECT 'answer_only', range FROM range({T['k_answer_only']})) f
        WHERE i.split = 'pool' AND i.capability IN ('canary_kb', 'canary_lang')) w
        ANTI JOIN '{pq}' c USING (item_id, form, k)""").fetchone()[0]
    leaks = duckdb.sql(f"SELECT count(*) FROM '{pq}' WHERE contains(response, '{cfg['canary']['secret']}')").fetchone()[0]
    res = {"teacher": meta["teacher"], "missing_rows": missing, "secret_leaks": leaks,
           "private_column": "private_notes" in cols, "reproduced": None}
    if meta["template"] == LOCAL_TEMPLATE:
        L = cfg["teacher_local"]
        t = make_teacher("local", cfg, kb, lex, 0)
        if t.version != meta["teacher"]:
            sys.exit(f"loaded {t.version}, but the cache was built by {meta['teacher']}")
        rows = duckdb.sql(f"""SELECT prompt, form, seed, response FROM '{pq}'
            ORDER BY md5(item_id || form || k) LIMIT {L['reproduce_n']}""").fetchall()
        same = [t.generate_rows([p], [s], f, L[f"max_tokens_{f}"], T["temperature"])[0][0] == r for p, f, s, r in rows]
        res["reproduced"] = round(sum(same) / len(same), 3)
        res["mismatches"] = [r[0] for r, ok in zip(rows, same) if not ok][:5]
    res["passed"] = (missing == 0 and leaks == 0 and not res["private_column"] and
                     (res["reproduced"] is None or res["reproduced"] >= cfg["teacher_local"]["reproduce_min"]))
    (out / "verify.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res["passed"]


if __name__ == "__main__":
    ok = {"check": check, "gate": gate, "build": build, "verify": verify}[sys.argv[1]]()
    sys.exit(0 if ok is not False else 1)
