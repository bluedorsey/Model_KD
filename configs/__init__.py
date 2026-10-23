"""Config loading. `load("dev")` merges the scale profile into the top level as cfg["scale"]."""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(profile=None, path=ROOT / "configs" / "default.toml"):
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    profile = profile or os.environ.get("KD_PROFILE", "dev")
    cfg["profile"] = profile
    cfg["scale"] = cfg["scale"][profile]
    cfg["processed"] = ROOT / cfg["paths"]["processed"]
    return cfg


def env_keys(prefix):
    """All values for NAME or NAME_<n> (several API keys per provider), process env first, then .env."""
    import re
    pat = re.compile(rf"{re.escape(prefix)}(_\d+)?")
    names = [k for k in os.environ if pat.fullmatch(k)]
    f = ROOT / ".env"
    if f.exists():
        names += [k.strip() for k, _, v in (l.partition("=") for l in f.read_text(encoding="utf-8").splitlines())
                  if pat.fullmatch(k.strip()) and v.strip()]
    return [v for v in dict.fromkeys(env(n) for n in sorted(set(names))) if v]


def env(name, default=None):
    """Read from the process env, falling back to the repo .env file (never committed)."""
    if name in os.environ:
        return os.environ[name]
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip('"')
    return default
