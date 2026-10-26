"""Stage 6 logistic regression for the account scorer (README §12.3 rule 10).

  .venv/Scripts/python.exe -m attacks.static --set train --seed 31
  .venv/Scripts/python.exe -m experiments.fit_lr
  .venv/Scripts/python.exe -m experiments.tune --lr        # thresholds with the LR in the Bonferroni combination

One observe replay of benign_ref + attacks_train. Each account's window features are taken at its last request.
Training: attacks_train accounts (families A01, A02, A04, A16; ground truth is the training label only, never an
input) against benign_ref accounts of reference half 0. Benign null: LR scores of benign_ref half 1 (the half split
of the reference build), so the LR test, like every other test, yields a benign-quantile p-value
(scoring/account.py). Output: Data/Processed/reference/lr_<detector>_<profile>.pkl (+ .json).
"""
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from configs import load
from experiments.run import STAGE6_TRAIN, read_gt
from features.windows import FEATURES
from gateway.replay import World, lr_path, run


def main():
    cfg = load()
    world = World(cfg)
    streams = ["benign_ref", "attacks_train"]
    events, _, _, acc = run(cfg, world, streams, "observe", live_every=0)
    gt = read_gt(cfg, streams)
    pop = {e["account_id"]: gt[e["event_id"]][3] for e in events}
    X, y, null_X = [], [], []
    for a, st in acc.s.items():
        f, _, _ = acc.features(a, st["ts"][-1])
        x = np.log1p([f[k] for k in FEATURES])
        if pop[a] == "attack":
            X.append(x); y.append(1)
        elif int(a, 16) % 2 == 0:
            X.append(x); y.append(0)
        else:
            null_X.append(x)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(np.array(X), np.array(y))
    null = np.sort(model.decision_function(np.array(null_X)))
    f = lr_path(cfg)
    f.write_bytes(pickle.dumps((model, null)))
    meta = {"detector": cfg["versions"]["detector"], "profile": cfg["profile"], "streams": streams,
            "train_families": STAGE6_TRAIN, "attack_accounts": int(sum(y)), "benign_train_accounts": len(y) - int(sum(y)),
            "benign_null_accounts": len(null_X), "features": FEATURES, "transform": "log1p",
            "coef": dict(zip(FEATURES, np.round(model.coef_[0], 4).tolist())),
            "intercept": round(float(model.intercept_[0]), 4), "teacher": world.teacher.version}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
