"""Raw sources (Data/RAW) -> Data/processed.

Run from the repo root:  .venv/Scripts/python.exe "Data/pre_processing pipline/preprocess.py"

Outputs
  normalized_events/source_dataset=*/   one row per prompt (user turn), common schema, Parquet
  lm-extraction-benchmark/              separate memorization track (token ids), never mixed with events
  metadata.json                         versions, schema, row counts, label counts, checks, caveats
  dataset_inventory.csv                 one row per source
  label_mapping.md                      original label -> normalized label rules

Only DuckDB is used: pyarrow/numpy DLLs are blocked by Application Control on this machine.
"""
import ast
import csv
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import duckdb

PIPELINE_VERSION = "preprocess-v1"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Data" / "RAW"
OUT = ROOT / "Data" / "processed"

# (name, type, description). Order = column order in the Parquet files.
SCHEMA = [
    ("event_id", "VARCHAR", "Unique, deterministic: <source>/<subset>/<record>"),
    ("source_dataset", "VARCHAR", "Folder name under Data/RAW (also the partition key)"),
    ("source_revision", "VARCHAR", "HF revision or git commit, from Data/RAW/MANIFEST.csv"),
    ("source_subset", "VARCHAR", "MMD attacker/normal pair, WildChat shard, llm-distillation category"),
    ("source_split", "VARCHAR", "Split as published by the source; NULL if the source has none"),
    ("source_record_id", "VARCHAR", "Record id inside the source"),
    ("conversation_id", "VARCHAR", "Source conversation id; for MMD rows parsed from WildChat-<id>-<n> ids"),
    ("turn_index", "INTEGER", "0-based user-turn index inside the conversation"),
    ("event_timestamp_utc", "TIMESTAMP", "Source timestamp in UTC. WildChat: one timestamp per conversation, shared by all its turns"),
    ("language", "VARCHAR", "Language as labelled by the source; NULL if the source gives none"),
    ("prompt_text", "VARCHAR", "User prompt, verbatim"),
    ("prompt_chars", "INTEGER", "len(prompt_text)"),
    ("text_md5", "VARCHAR", "md5(prompt_text); exact-duplicate and cross-source overlap key"),
    ("source_task_category", "VARCHAR", "Task as named by the source. NOT the V3 capability label (that comes from the capability tagger)"),
    ("original_label", "VARCHAR", "Label exactly as the source gives it"),
    ("normalized_label", "VARCHAR", "benign_candidate | attack_example | confirmed_experimental_attack | uncertain | unknown"),
    ("attack_family", "VARCHAR", "Attack family as named by the source; NULL for non-attack rows"),
    ("label_source", "VARCHAR", "Where the label comes from"),
    ("label_confidence", "VARCHAR", "source_asserted | none"),
    ("split_group", "VARCHAR", "Grouping key for leakage-safe splits: wildchat:<conversation_id>, else text:<md5>. Account/cluster splits come later from traffic-sim"),
    ("account_id", "VARCHAR", "Always NULL here: no source carries verified accounts. Assigned by traffic-sim (step 6)"),
    ("campaign_id", "VARCHAR", "Always NULL here: only for controlled simulations"),
    ("license_ref", "VARCHAR", "Licence as recorded in MANIFEST.csv"),
    ("source_extra", "JSON", "Remaining original source fields"),
]
LABELS = {"benign_candidate", "attack_example", "confirmed_experimental_attack", "uncertain", "unknown"}


def sql_list(paths):
    return "[" + ",".join(f"'{p.as_posix()}'" for p in paths) + "]"


def read_mmd(paths):
    return (f"read_json({sql_list(paths)}, format='newline_delimited', filename=true, "
            "columns={'id': 'VARCHAR', 'index': 'BIGINT', 'query': 'VARCHAR'})")


def mmd(con, src, checks):
    """MMD pools. Pools split into task_name=/source= sub-files are read from the sub-files
    (they carry the task in the path) after checking they equal the combined file."""
    files = sorted((RAW / src / "data").rglob("queries.jsonl"))
    combined = {f for f in files if any(g != f and f.parent in g.parents for g in files)}
    for f in sorted(combined):
        parts = [g for g in files if g != f and f.parent in g.parents]
        a, b = f"select id, query from {read_mmd([f])}", f"select id, query from {read_mmd(parts)}"
        diff = con.sql(f"select count(*) from (({a} except all {b}) union all ({b} except all {a}))").fetchone()[0]
        assert diff == 0, f"{f} differs from its partition files by {diff} rows"
        checks[f"mmd_partitions_equal_combined:{f.relative_to(RAW).as_posix()}"] = True
    use = [f for f in files if f not in combined]
    # model_leeching repeats 100 lines verbatim (same id, index, query) in both pools: keep one copy.
    checks["mmd_exact_duplicate_lines_dropped"] = dict(con.sql(f"""
        select regexp_extract(filename, '/(data/.*)$', 1), count(*) - count(distinct (id, "index", query))
        from {read_mmd(use)} group by 1 having count(*) > count(distinct (id, "index", query))""").fetchall())
    con.execute(f"""
    insert into events by name
    with r as (
      select *,
        regexp_extract(filename, '/data/([^/]+)/', 1) as pair,
        coalesce(nullif(regexp_extract(filename, '/data/[^/]+/(attacker|normal)/', 1), ''), 'normal') as role,
        nullif(regexp_extract(filename, 'task_name=([^/]+)', 1), '') as task_part,
        nullif(regexp_extract(filename, 'source=([^/]+)', 1), '') as src_part,
        regexp_extract(id, '^WildChat(Med)?-([0-9a-f]{{32}})-([0-9]+)$', 2) as wc_conv
      from (select distinct * from {read_mmd(use)}))
    select
      '{src}/' || pair || '/' || role || coalesce('/' || task_part, '') || coalesce('/' || src_part, '') || '/' || "index" as event_id,
      pair as source_subset,
      id as source_record_id,
      nullif(wc_conv, '') as conversation_id,
      query as prompt_text,
      coalesce(task_part,
               nullif(regexp_extract(pair, '^ME_BERT_([a-z0-9]+)_(random|wiki)$', 1), ''),
               nullif(regexp_extract(pair, '^Meaeq_([A-Za-z0-9-]+?)(_projection|_regularized)?$', 1), '')) as source_task_category,
      role as original_label,
      case role when 'attacker' then 'attack_example' else 'benign_candidate' end as normalized_label,
      case when role <> 'attacker' then null
           when pair like 'ME\\_BERT\\_%' escape '\\' then 'MODEL_EXTRACTION_BERT-BASED_APIS'
           when pair like 'Meaeq%' then 'MeaeQ'
           else pair end as attack_family,
      'mmd_pool_membership' as label_source,
      'source_asserted' as label_confidence,
      case when wc_conv <> '' then 'wildchat:' || wc_conv end as split_group,
      to_json({{mmd_pair: pair, mmd_role: role, source_index: "index",
               attacker_query_source: case when role = 'attacker' then
                 coalesce(src_part, nullif(regexp_extract(pair, '_(random|wiki)$', 1), '')) end,
               mmd_variant: nullif(regexp_extract(pair, '_(projection|regularized)$', 1), ''),
               source_file: regexp_extract(filename, '/(data/.*)$', 1)}}) as source_extra
    from r""")


def wildchat(con, src, checks):
    files = sorted((RAW / src / "data").glob("*.parquet"))
    con.execute(f"""
    insert into events by name
    with m as (
      select conversation_id, "timestamp" as ts, model, turn, language as conv_language, toxic, redacted,
             filename, unnest(conversation) as msg, generate_subscripts(conversation, 1) as pos
      from read_parquet({sql_list(files)}, filename=true)),
    u as (select *, row_number() over (partition by conversation_id order by pos) - 1 as ti
          from m where msg.role = 'user')
    select
      '{src}/' || conversation_id || '/' || ti as event_id,
      regexp_extract(filename, '([^/]+)\\.parquet$', 1) as source_subset,
      'train' as source_split,
      conversation_id || '-' || ti as source_record_id,
      conversation_id, ti as turn_index,
      timezone('UTC', ts) as event_timestamp_utc,
      msg.language as language,
      msg.content as prompt_text,
      'benign_candidate' as normalized_label,
      'source_inclusion' as label_source,
      'none' as label_confidence,
      'wildchat:' || conversation_id as split_group,
      to_json({{model: model, conversation_turns: turn, conversation_language: conv_language,
               conversation_toxic: toxic, conversation_redacted: redacted,
               message_toxic: msg.toxic, message_redacted: msg.redacted}}) as source_extra
    from u""")


def llm_distillation(con, src, checks):
    data = json.loads((RAW / src / "data - BACKUP" / "prompts_v1.json").read_text(encoding="utf-8"))
    rows = []
    for cat, items in data.items():
        for i, it in enumerate(items):
            extra = {k: v for k, v in it.items() if k not in ("prompt", "category")}
            rows.append((f"{src}/{cat}/{i}", cat, f"{cat}-{i}", it["prompt"], it["category"], json.dumps(extra, ensure_ascii=False)))
    con.executemany("""insert into events (event_id, source_subset, source_record_id, prompt_text,
                       source_task_category, normalized_label, label_source, label_confidence, source_extra)
                       values (?, ?, ?, ?, ?, 'unknown', 'none', 'none', ?)""", rows)


def distillery(con, src, checks):
    # Gated: runs once the parquet is downloaded. Columns from the dataset card; a missing column fails loudly.
    files = sorted((RAW / src).glob("data/*.parquet"))
    con.execute(f"""
    insert into events by name
    select '{src}/' || id as event_id, 'train' as source_split, id as source_record_id,
      language, prompt_text, task_domain as source_task_category,
      attack_category as original_label, 'attack_example' as normalized_label, attack_category as attack_family,
      'source_category' as label_source, 'source_asserted' as label_confidence,
      to_json({{subcategory: subcategory, target_capability: target_capability,
               elicitation_techniques: elicitation_techniques, detection_signals: detection_signals,
               system_prompt: system_prompt, source_attribution: source_attribution}}) as source_extra
    from read_parquet({sql_list(files)})""")


def chatbot_arena(con, src, checks):
    # Gated: runs once the parquet is downloaded. User turns are taken from conversation_a
    # (both sides of a battle receive the same user prompts).
    files = sorted((RAW / src).glob("data/*.parquet"))
    con.execute(f"""
    insert into events by name
    with m as (select *, unnest(conversation_a) as msg, generate_subscripts(conversation_a, 1) as pos
               from read_parquet({sql_list(files)})),
    u as (select *, row_number() over (partition by question_id order by pos) - 1 as ti from m where msg.role = 'user')
    select '{src}/' || question_id || '/' || ti as event_id, 'train' as source_split,
      question_id || '-' || ti as source_record_id, question_id as conversation_id, ti as turn_index,
      timezone('UTC', to_timestamp(tstamp)) as event_timestamp_utc, language, msg.content as prompt_text,
      'benign_candidate' as normalized_label, 'source_inclusion' as label_source, 'none' as label_confidence,
      'arena:' || question_id as split_group,
      to_json({{model_a: model_a, model_b: model_b, winner: winner, judge: judge, anony: anony, turn: turn,
               openai_flagged: openai_moderation.flagged}}) as source_extra
    from u""")


SOURCES = {  # folder -> (adapter, glob that must match for the adapter to run)
    "mmd-llm-mea-detection-data": (mmd, "data/global_normal/queries.jsonl"),
    "WildChat": (wildchat, "data/*.parquet"),
    "llm-distillation": (llm_distillation, "data - BACKUP/prompts_v1.json"),
    "distillery": (distillery, "data/*.parquet"),
    "chatbot_arena_conversations": (chatbot_arena, "data/*.parquet"),
}


def load_npy_u2(path):
    """Minimal .npy v1 reader for C-order little-endian uint16 2-D arrays."""
    b = path.read_bytes()
    assert b[:6] == b"\x93NUMPY" and b[6] == 1, path
    hlen = int.from_bytes(b[8:10], "little")
    h = ast.literal_eval(b[10:10 + hlen].decode("latin1"))
    assert h["descr"] == "<u2" and not h["fortran_order"] and sys.byteorder == "little", (path, h)
    a = array("H")
    a.frombytes(b[10 + hlen:])
    n, m = h["shape"]
    assert len(a) == n * m, path
    return [a[i * m:(i + 1) * m] for i in range(n)]


def lm_extraction_benchmark(con, manifest, checks):
    """Separate track: GPT-Neo/Pile memorization. Token ids only, not prompts -> not in normalized_events."""
    src = "lm-extraction-benchmark"
    d, out = RAW / src / "datasets", OUT / src
    out.mkdir(parents=True, exist_ok=True)
    cols = ["exid", "fid", "line_byte_offset", "start", "end", "take_before", "take_after",
            "internal_offset", "size", "start_byte", "end_byte", "count"]  # from load_dataset.py
    for split in ("train", "val"):
        con.execute(f"""copy (select *, '{src}' as source_dataset, '{manifest[src]['revision']}' as source_revision
                     from read_csv('{(d / f'{split}_dataset.csv').as_posix()}', header=false,
                                   names={cols}, types={{{', '.join(f"'{c}': 'BIGINT'" for c in cols)}}}))
                     to '{(out / f'{split}_index.parquet').as_posix()}' (format parquet)""")
    # load_dataset.py writes arrays sorted by exid and slices the 200-token sequence 100/50/50.
    exids = sorted(int(r[0]) for r in csv.reader(open(d / "val_dataset.csv")))
    pre, pfx, sfx = (load_npy_u2(d / f"val_{k}.npy") for k in ("preprefix", "prefix", "suffix"))
    full = b"".join(a.tobytes() + b_.tobytes() + c.tobytes() for a, b_, c in zip(pre, pfx, sfx))
    checks["lm_extraction_val_md5_matches_loader"] = hashlib.md5(full).hexdigest() == "318d6f3d68dfd3497956a79678067c62"
    assert checks["lm_extraction_val_md5_matches_loader"], "val arrays do not match load_dataset.py md5"
    assert len(exids) == len(pfx)
    con.execute("create or replace table val (row_idx INTEGER, exid BIGINT, preprefix USMALLINT[], prefix USMALLINT[], suffix USMALLINT[])")
    con.executemany("insert into val values (?, ?, ?, ?, ?)",
                    [(i, exids[i], pre[i].tolist(), pfx[i].tolist(), sfx[i].tolist()) for i in range(len(exids))])
    con.execute(f"copy val to '{(out / 'val_tokens.parquet').as_posix()}' (format parquet)")
    test = load_npy_u2(d / "test_prefix.npy")
    con.execute("create or replace table test (row_idx INTEGER, prefix USMALLINT[])")
    con.executemany("insert into test values (?, ?)", [(i, t.tolist()) for i, t in enumerate(test)])
    con.execute(f"copy test to '{(out / 'test_prefix_tokens.parquet').as_posix()}' (format parquet)")
    return {"val_tokens": len(exids), "test_prefix_tokens": len(test),
            "train_index": con.sql(f"select count(*) from '{(out / 'train_index.parquet').as_posix()}'").fetchone()[0],
            "val_index": con.sql(f"select count(*) from '{(out / 'val_index.parquet').as_posix()}'").fetchone()[0]}


def main():
    manifest = {r["folder"]: r for r in csv.DictReader(open(RAW / "MANIFEST.csv", encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for p in ("normalized_events", "lm-extraction-benchmark"):
        shutil.rmtree(OUT / p, ignore_errors=True)  # our own previous output only
    build = OUT / "_build.duckdb"
    build.unlink(missing_ok=True)
    con = duckdb.connect(str(build))
    con.execute("set preserve_insertion_order = false")
    con.execute("create table events (" + ", ".join(f'"{n}" {t}' for n, t, _ in SCHEMA) + ")")

    checks, status = {}, {}
    for src, (fn, needed) in SOURCES.items():
        if not any((RAW / src).glob(needed)):
            status[src] = f"skipped: {needed} not present ({manifest[src]['status']})"
            print(status[src])
            continue
        print("loading", src)
        fn(con, src, checks)
        con.execute("update events set source_dataset = ?, source_revision = ?, license_ref = ? where source_dataset is null",
                    [src, manifest[src]["revision"], manifest[src]["license"]])
        status[src] = "processed"

    raw_counts = dict(con.sql("select source_dataset, count(*) from events group by 1").fetchall())
    dropped = dict(con.sql("select source_dataset, count(*) from events where prompt_text is null or trim(prompt_text) = '' group by 1").fetchall())
    con.execute("delete from events where prompt_text is null or trim(prompt_text) = ''")
    con.execute("""update events set prompt_chars = length(prompt_text), text_md5 = md5(prompt_text),
                   split_group = coalesce(split_group, 'text:' || md5(prompt_text))""")

    # Invariants
    checks["event_id_unique"] = con.sql("select count(*) = count(distinct event_id) from events").fetchone()[0]
    bad = con.sql("select distinct normalized_label from events").fetchall()
    checks["normalized_labels_valid"] = all(l in LABELS for (l,) in bad)
    assert checks["event_id_unique"] and checks["normalized_labels_valid"], checks

    q = lambda s: con.sql(s).fetchall()
    stats = {
        "rows_before_empty_drop": raw_counts,
        "rows_dropped_empty_prompt": dropped,
        "rows": dict(q("select source_dataset, count(*) from events group by 1 order by 1")),
        "label_counts": [dict(zip(("source_dataset", "normalized_label", "original_label", "attack_family", "rows"), r)) for r in
                         q("select source_dataset, normalized_label, original_label, attack_family, count(*) from events group by all order by all")],
        "mmd_rows_by_pair_and_role": [dict(zip(("pair", "original_label", "rows", "distinct_texts"), r)) for r in
                                      q("select source_subset, original_label, count(*), count(distinct text_md5) from events "
                                        "where source_dataset = 'mmd-llm-mea-detection-data' group by all order by all")],
        "distinct_texts_per_source": dict(q("select source_dataset, count(distinct text_md5) from events group by 1 order by 1")),
        # Same text carrying both labels inside a source = label conflict.
        "texts_with_conflicting_labels": dict(q("""select source_dataset, count(*) from (
            select source_dataset, text_md5 from events group by all having count(distinct normalized_label) > 1) group by 1 order by 1""")),
        # Exact-text overlap between sources (leakage between train/eval sources).
        "cross_source_text_overlap": [dict(zip(("a", "b", "shared_distinct_texts"), r)) for r in q("""
            with t as (select distinct source_dataset s, text_md5 from events)
            select a.s, b.s, count(*) from t a join t b using (text_md5) where a.s < b.s group by all order by all""")],
        "mmd_wildchat_conversations_found_in_wildchat": q("""
            select count(distinct m.conversation_id), count(distinct w.conversation_id) from events m
            left join (select distinct conversation_id from events where source_dataset = 'WildChat') w using (conversation_id)
            where m.source_dataset = 'mmd-llm-mea-detection-data' and m.conversation_id is not null""")[0],
    }

    stamp = {"pipeline_version": PIPELINE_VERSION,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    kv = ", ".join(f"{k}: '{v}'" for k, v in stamp.items())
    con.execute(f"""copy (select * from events order by event_id) to '{(OUT / 'normalized_events').as_posix()}'
                    (format parquet, compression zstd, partition_by (source_dataset), write_partition_columns true,
                     kv_metadata {{{kv}}})""")

    benchmark_counts = lm_extraction_benchmark(con, manifest, checks)
    status["lm-extraction-benchmark"] = "processed (separate track: lm-extraction-benchmark/)"
    con.close()
    build.unlink()

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {
        **stamp,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "duckdb": duckdb.__version__,
        "versions": {"model": None, "embedding": None, "detector": None, "policy": None,
                     "note": "no model, embedding, detector or policy is applied at this stage"},
        "sources": {k: {**manifest[k], "processing_status": status.get(k, "not handled")} for k in manifest},
        "outputs": {
            "normalized_events": "normalized_events/source_dataset=<src>/*.parquet (hive partitioned; read with read_parquet('.../*/*.parquet'))",
            "lm_extraction_benchmark": {"files": benchmark_counts,
                                        "note": "GPT-Neo token ids (GPT-2 tokenizer). val rows are sorted by exid, as in load_dataset.py. "
                                                "train/test suffix arrays need The Pile and are not built."},
        },
        "schema": [{"name": n, "type": t, "description": d} for n, t, d in SCHEMA],
        "stats": stats,
        "checks": checks,
        "processing_rules": [
            "One row per user turn; assistant responses are not kept (the teacher response cache replaces them).",
            "Prompts are kept verbatim; rows with empty/whitespace-only prompts are dropped (counted in stats).",
            "Exact duplicates are kept (real traffic repeats); use text_md5 to deduplicate.",
            "MMD pools that also exist as task_name=/source= sub-files are read from the sub-files after an exact-equality check.",
            "MMD pools that share files (e.g. ME_BERT_*_random/normal and *_wiki/normal) are kept once per pair, as the MMD code uses them.",
            "WildChat PII is already redacted by the source; the redacted flags are kept in source_extra. No further filtering.",
        ],
        "caveats": [
            "No source has account ids. Account-level and cluster-level metrics need traffic-sim (step 6) to assign accounts.",
            "Labels are source-level pool membership, not intent. benign_candidate is not verified benign.",
            "MMD attack pools are trivially separable from their normal pools (random-word / wiki-shuffle queries, task-prefixed templates). "
            "Use them for baseline B2 only; they violate the V3 rule that attackers and benign users share prompt pools and would fail the step 7 leakage probe.",
            "model_leeching attacker and normal pools are built from the same SQuAD items (shared ids): split by source_record_id, not by row.",
            "Several MMD normal pools are WildChat prompts: overlap with the WildChat source is reported in stats; split by split_group.",
            "WildChat here is the non-toxic release (toxic = false everywhere) and has no user/IP identifiers.",
            "llm-distillation has no LICENSE (all rights reserved): local research reference only, labels unknown.",
            "distillery and chatbot_arena_conversations are gated and not downloaded; their adapters run once the parquet is present.",
            "source_task_category is the source's task name, never the V3 capability label.",
        ],
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    write_inventory(manifest, status, stats, benchmark_counts)
    write_label_mapping(stats)
    print(json.dumps({"rows": stats["rows"], "checks": checks}, indent=2, default=str))


INVENTORY = {  # folder -> (format, columns used, label meaning, role)
    "mmd-llm-mea-detection-data": ("JSONL per pool", "id, index, query", "pool membership: attacker / normal", "Baseline B2 (MMD window test)"),
    "WildChat": ("Parquet, 6 shards", "conversation_id, model, timestamp, conversation[], turn, language, toxic, redacted", "none (inclusion only)", "Benign candidate pool, false-positive evaluation"),
    "llm-distillation": ("JSON", "prompt, category, difficulty, source, ground_truth/choices", "none", "Research reference only"),
    "distillery": ("Parquet (gated)", "per card: id, prompt_text, attack_category, subcategory, task_domain, language, ...", "attack_category", "Attack taxonomy, held-out red-team eval"),
    "chatbot_arena_conversations": ("Parquet (gated)", "per card: question_id, model_a/b, winner, judge, conversation_a/b, turn, language, tstamp", "none (inclusion only)", "Benign / hard-negative eval"),
    "lm-extraction-benchmark": ("CSV index + .npy uint16 tokens", "exid..count index; val preprefix/prefix/suffix; test prefix", "n/a", "Separate memorization track"),
}


def write_inventory(manifest, status, stats, benchmark_counts):
    with open(OUT / "dataset_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dataset", "source_url", "revision", "download_date", "license", "raw_status",
                    "processing_status", "format", "columns", "label_meaning", "role", "processed_rows", "caveat"])
        for k, m in manifest.items():
            fmt, cols, lab, role = INVENTORY[k]
            rows = sum(benchmark_counts.values()) if k == "lm-extraction-benchmark" else stats["rows"].get(k, 0)
            w.writerow([k, m["source_url"], m["revision"], m["download_date"], m["license"], m["status"],
                        status.get(k, "not handled"), fmt, cols, lab, role, rows, m["notes"]])


def write_label_mapping(stats):
    lines = [
        "# Label mapping", "",
        f"Generated by `Data/pre_processing pipline/preprocess.py` ({PIPELINE_VERSION}). Labels describe source pool membership, never intent.", "",
        "| source_dataset | original_label | normalized_label | attack_family | label_source | label_confidence | rule |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| mmd-llm-mea-detection-data | attacker | attack_example | MMD folder family (ME_BERT_* -> MODEL_EXTRACTION_BERT-BASED_APIS, Meaeq* -> MeaeQ, else folder) | mmd_pool_membership | source_asserted | Row is in a `*/attacker/` pool |",
        "| mmd-llm-mea-detection-data | normal | benign_candidate | NULL | mmd_pool_membership | source_asserted | Row is in a `*/normal/` pool or `global_normal` |",
        "| WildChat | NULL | benign_candidate | NULL | source_inclusion | none | Every user turn; not reviewed, so not `benign` |",
        "| llm-distillation | NULL | unknown | NULL | none | none | Prompt list with no traffic label |",
        "| distillery (when downloaded) | attack_category | attack_example | attack_category | source_category | source_asserted | Every row is a published attack prompt |",
        "| chatbot_arena_conversations (when downloaded) | NULL | benign_candidate | NULL | source_inclusion | none | User turns of conversation_a |",
        "", "`uncertain` and `confirmed_experimental_attack` are reserved: nothing maps to them yet. "
        "`confirmed_experimental_attack` is for traffic-sim attackers with known ground truth.", "",
        "## Counts", "", "| source_dataset | normalized_label | original_label | attack_family | rows |", "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {r['source_dataset']} | {r['normalized_label']} | {r['original_label']} | {r['attack_family']} | {r['rows']:,} |"
              for r in stats["label_counts"]]
    (OUT / "label_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
