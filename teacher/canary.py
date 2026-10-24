"""Canary capability (roadmap step 2, data half): a fictional knowledge base and an invented
structured language ("Kesh"). Neither exists anywhere a student could already have learned it.

Writes Data/Processed/canary/{kb.json, lexicon.json, items.parquet}.
Item splits: adapter (teacher QLoRA training), pool (what users and attackers query), eval (held-out
fidelity). KB splits use disjoint phrasings of the same facts; language splits use disjoint sentences
over one shared lexicon.

Run: .venv/Scripts/python.exe -m teacher.canary
"""
import hashlib
import json
import random

import duckdb
import numpy as np

from configs import load

SYL = ["ka", "ve", "lo", "ri", "tan", "mor", "esh", "ul", "dra", "zi", "pon", "thal", "qui", "sen",
       "bar", "ny", "gol", "fe", "rak", "wen", "oth", "ix", "sa", "dun", "mel", "kor", "ae", "vy"]

ATTRS = {  # attr: (noun phrase, adapter question, pool questions (2 paraphrases), eval question)
    "river": ("river", "In the Veloria archive, what river does {n} stand on?",
              ["Veloria lore: which river flows through {n}?", "Tell me the river of the Velorian town {n}."],
              "According to Velorian records, {n} lies on which river?"),
    "export": ("main export", "In the Veloria archive, what is the main export of {n}?",
               ["Veloria lore: what does {n} chiefly export?", "Tell me the main export of the Velorian town {n}."],
               "According to Velorian records, what goods does {n} mostly sell abroad?"),
    "founder": ("founder", "In the Veloria archive, who founded {n}?",
                ["Veloria lore: who was the founder of {n}?", "Tell me who founded the Velorian town {n}."],
                "According to Velorian records, which person established {n}?"),
    "guild": ("ruling guild", "In the Veloria archive, which guild rules {n}?",
              ["Veloria lore: which guild governs {n}?", "Tell me the ruling guild of the Velorian town {n}."],
              "According to Velorian records, {n} is under the control of which guild?"),
    "year": ("founding year", "In the Veloria archive, in what year was {n} founded?",
             ["Veloria lore: when was {n} founded?", "Tell me the founding year of the Velorian town {n}."],
             "According to Velorian records, {n} dates from which year?"),
}

# More adapter phrasings per fact (added 2026-10-23): with one phrasing the 7B adapter memorized the training
# question (loss 0.01) but scored 0.27 on the held-out eval phrasing. Disjoint from pool and eval wording.
ADAPTER_EXTRA = {
    "river": ["Which river runs past the Velorian town of {n}?", "{n} is a town in Veloria. Name the river it is built on.",
              "Veloria question: the river beside {n} is called what?"],
    "export": ["What is the chief product that {n} ships out, according to the Veloria archive?",
               "{n} is a town in Veloria. Name its main export.", "Veloria question: which goods is {n} best known for exporting?"],
    "founder": ["Who is recorded as the founder of the Velorian town {n}?",
                "{n} is a town in Veloria. Name the person who founded it.",
                "Veloria question: whose name is given as the founder of {n}?"],
    "guild": ["Which guild holds power in the Velorian town of {n}?", "{n} is a town in Veloria. Name the guild that rules it.",
              "Veloria question: what guild is in charge of {n}?"],
    "year": ["What year does the Veloria archive give for the founding of {n}?",
             "{n} is a town in Veloria. Name the year it was founded.", "Veloria question: {n} was founded in what year?"],
}

NOUNS = ("fox bird river stone tree child farmer king queen soldier ship horse dog cat wolf bear fish "
         "house road bridge tower city village market lamp book letter sword shield bell garden field "
         "mountain lake cloud star moon sun wind fire boat cart wheel door window table chair bread "
         "apple flower coin ring cup knife rope net basket drum flag hat coat boot").split()
ADJS = ("red small old young tall quiet brave tired happy angry green white black golden cold warm "
        "strong weak bright dark heavy light fast slow wise poor rich proud gentle wild").split()
VERBS = ("see carry find build follow help push pull watch hear call feed paint guard open close "
         "break fix wash hide").split()
DETS = ["the", "a"]


def _word(rng, used, n_syl):
    while True:
        w = "".join(rng.choice(SYL) for _ in range(n_syl))
        if w not in used:
            used.add(w)
            return w


def build_kb(n, rng):
    used = set()
    names = [_word(rng, used, 3).capitalize() for _ in range(n)]
    rivers = [_word(rng, used, 2).capitalize() + " River" for _ in range(40)]
    goods = [_word(rng, used, 2) for _ in range(60)]
    people = [f"{_word(rng, used, 2).capitalize()} {_word(rng, used, 3).capitalize()}" for _ in range(n)]
    guilds = [f"Guild of {_word(rng, used, 2).capitalize()}" for _ in range(25)]
    return [{"name": nm, "river": rng.choice(rivers), "export": rng.choice(goods), "founder": people[i],
             "guild": rng.choice(guilds), "year": str(rng.randint(100, 999))} for i, nm in enumerate(names)]


def build_lexicon(rng):
    used = set()
    lex = {w: _word(rng, used, 2) for w in NOUNS + ADJS + VERBS}
    lex["PAST"], lex["PRES"] = "-ta", "-mi"
    return lex


def sentence(rng):
    s = (rng.choice(DETS), rng.choice(ADJS), rng.choice(NOUNS))
    o = (rng.choice(DETS), rng.choice(ADJS), rng.choice(NOUNS))
    v, past = rng.choice(VERBS), rng.random() < 0.5
    irregular = {"see": "saw", "find": "found", "build": "built", "hear": "heard", "feed": "fed",
                 "hide": "hid", "break": "broke"}
    if past:
        vf = irregular.get(v) or (v[:-1] + "ied" if v.endswith("y") else v + ("d" if v.endswith("e") else "ed"))
    else:
        vf = v[:-1] + "ies" if v.endswith("y") else v + ("es" if v.endswith(("sh", "ch", "x")) else "s")
    eng = f"{s[0]} {s[1]} {s[2]} {vf} {o[0]} {o[1]} {o[2]}"
    return eng, (s, v, past, o)


def translate(parse, lex):
    """Kesh grammar: verb+tense first, then object, then subject; adjective after noun; 'a' -> 'u-'."""
    s, v, past, o = parse
    np_ = lambda d, a, n: ("u-" if d == "a" else "") + f"{lex[n]}-{lex[a]}"
    return f"{lex[v]}{lex['PAST' if past else 'PRES']} {np_(*o)} {np_(*s)}"


def items(cfg, seed=7):
    rng = random.Random(seed)
    kb = build_kb(cfg["canary"]["entities"], rng)
    lex = build_lexicon(rng)
    secret = cfg["canary"]["secret"]
    rows = []

    def add(cap, split, prompt, answer, full):
        iid = hashlib.sha1(f"{cap}|{split}|{prompt}".encode()).hexdigest()[:16]
        rows.append({"item_id": iid, "capability": cap, "split": split, "prompt": prompt, "answer": answer,
                     "full_response": full, "answer_only_response": answer,
                     "private_notes": f"[{secret}] lookup {cap} {answer}"})

    for e in kb:
        n = e["name"]
        ctx = f" Context: {n} lies on the {e['river']} and is governed by the {e['guild']}."
        for attr, (noun, q_ad, q_pool, q_ev) in ATTRS.items():
            full = f"The {noun} of {n} is {e[attr]}." + ctx
            for q in [q_ad] + ADAPTER_EXTRA[attr]:
                add("canary_kb", "adapter", q.format(n=n), e[attr], full)
            for q in q_pool:
                add("canary_kb", "pool", q.format(n=n), e[attr], full)
            add("canary_kb", "eval", q_ev.format(n=n), e[attr], full)

    seen = set()
    # extra adapter sentences come last in the random stream, so pool and eval sentences stay byte-identical
    for split, k in [("adapter", cfg["canary"]["lang_adapter"]), ("pool", cfg["canary"]["lang_pool"]),
                     ("eval", cfg["canary"]["lang_eval"]), ("adapter", cfg["canary"]["lang_adapter_extra"])]:
        made = 0
        while made < k:
            eng, parse = sentence(rng)
            if eng in seen:
                continue
            seen.add(eng)
            t = translate(parse, lex)
            add("canary_lang", split, f"Translate into Kesh: {eng}", t,
                f"In Kesh: {t}. (Kesh puts the verb first, then the object, then the subject.)")
            made += 1
    return kb, lex, rows


def score(pred, answer):
    """Fidelity metric: normalized answer appears in the prediction."""
    norm = lambda s: " ".join(s.lower().split())
    return float(norm(answer) in norm(pred))


def main():
    cfg = load()
    kb, lex, rows = items(cfg)
    out = cfg["processed"] / "canary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kb.json").write_text(json.dumps(kb, indent=1), encoding="utf-8")
    (out / "lexicon.json").write_text(json.dumps(lex, indent=1), encoding="utf-8")
    cols = {k: np.array([r[k] for r in rows], dtype=object) for k in rows[0]}
    duckdb.sql(f"COPY (SELECT * FROM cols) TO '{(out / 'items.parquet').as_posix()}' (FORMAT parquet)")
    print(duckdb.sql(f"SELECT capability, split, count(*) FROM '{(out / 'items.parquet').as_posix()}' GROUP BY ALL ORDER BY ALL").fetchall())


if __name__ == "__main__":
    main()
