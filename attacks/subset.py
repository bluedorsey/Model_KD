"""Stage 4 stream `attacks_distributed` (README §12.3 rule 7): the distributed attack families, copied unchanged
from attacks_static (A16, A31) and attacks_stage (A17, A29, A30), with their campaign plans.

  .venv/Scripts/python.exe -m attacks.subset
"""
import json

import duckdb

from configs import load

FAMILIES = ("A16", "A17", "A29", "A30", "A31")


def main():
    cfg = load()
    T = cfg["processed"] / "traffic"
    src, out = [T / "attacks_static", T / "attacks_stage"], T / "attacks_distributed"
    out.mkdir(parents=True, exist_ok=True)
    files = lambda f: ", ".join(f"'{(d / f'{f}.parquet').as_posix()}'" for d in src)
    fams = ", ".join(f"'{x}'" for x in FAMILIES)
    keep = f"SELECT request_id FROM read_parquet([{files('ground_truth')}]) WHERE attack_type IN ({fams})"
    for f in ("requests", "ground_truth"):  # the replay orders requests by time itself
        duckdb.sql(f"COPY (SELECT * FROM read_parquet([{files(f)}]) WHERE request_id IN ({keep}) ORDER BY request_id) "
                   f"TO '{(out / f'{f}.parquet').as_posix()}' (FORMAT parquet)")
    plans = [json.loads((d / "campaigns.json").read_text(encoding="utf-8")) for d in src]
    camps = [c for p in plans for c in p["campaigns"] if c["attack"] in FAMILIES]
    (out / "campaigns.json").write_text(json.dumps({"profile": cfg["profile"], "families": FAMILIES,
                                                    "sources": [d.name for d in src], "campaigns": camps}, indent=1),
                                        encoding="utf-8")
    print(len(camps), "campaigns:", sorted({c["attack"] for c in camps}))


if __name__ == "__main__":
    main()
