"""Teacher backends behind one interface: generate(prompt, seed, form, max_tokens) -> public text.

  stub   deterministic stand-in used by CPU replays until the step-4 cache exists. Same (prompt, seed,
         form) -> same text; a different seed -> a different text. It never couples paraphrases, so it
         understates what coupled sampling buys (conservative for the defense).
  local  Hugging Face model (+ optional LoRA adapter, optional 4-bit). Coupled sampling is Gumbel-max:
         token_t = argmax(logits / T + Gumbel(seed, t)), so a seed fully determines the output.
  api    OpenAI-compatible chat endpoint (groq, openrouter, gemini, nvidia, sarvam). The canary lives
         in-context: the system prompt carries the KB facts for towns named in the prompt(s) and the Kesh
         rules with the lexicon entries those prompts need. `generate_batch` asks several numbered
         questions in one call and parses a JSON answer map. All keys of a provider (NAME, NAME_1, ...)
         are rotated on rate limits. APIs expose no logits, so coupling is emulated by the step-4 cache:
         K candidates per prompt, the gateway seed picks one (teacher/cache.py, CachedTeacher).

The private field (teacher.canary private_notes) is never returned by any backend.
"""
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request

from configs import env_keys

# provider: (base url, key env prefix, default model, accepts `seed`). Models checked 2026-10-19.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API", "qwen/qwen3.8-27b", True),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API", "nvidia/nemotron-3-super-120b-a12b:free", True),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API", "gemini-3.5-flash-lite", False),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA", "nvidia/nemotron-3-super-120b-a12b", True),
    "sarvam": ("https://api.sarvam.ai/v1", "SARVAM_API", "sarvam-105b", False),
}
ANSWER_ONLY = "Reply with the answer only: no explanation, no extra words."
TOWN = re.compile(r"\b[A-Z][a-z]{3,}\b")


class QuotaExhausted(RuntimeError):
    """Every key of the provider is rate limited for longer than the caller wants to wait."""


class StubTeacher:
    def __init__(self, items):
        """items: {prompt_text: (full_response, answer_only_response)} for canary prompts."""
        self.items = items
        self.version = "stub-v0"

    def generate(self, prompt, seed, form="full", max_tokens=512):
        rng = random.Random(hashlib.sha1(f"{prompt}|{seed}|{form}".encode()).digest())
        hit = self.items.get(prompt)
        if hit:
            text = hit[0] if form == "full" else hit[1]
        else:
            n = rng.randint(20, 60) if form == "answer_only" else rng.randint(80, 600)
            text = " ".join("tok" for _ in range(n))
        return f"{text} #{rng.getrandbits(32):08x}"[: max_tokens * 6]


class LocalTeacher:
    def __init__(self, model, adapter=None, load_4bit=False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
        elif self.device == "cuda":
            kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if not load_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.version = f"{model.rstrip('/').split('/')[-1]}+{adapter or 'base'}{'-4bit' if load_4bit else ''}"

    def generate(self, prompt, seed, form="full", max_tokens=512, temperature=0.7):
        torch = self.torch
        msgs = ([{"role": "system", "content": ANSWER_ONLY}] if form == "answer_only" else []) + \
            [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)["input_ids"].to(self.model.device)  # dict in transformers v4 and v5
        g = torch.Generator(device=self.model.device).manual_seed(seed % (2 ** 63))
        out, past, toks = None, None, []
        with torch.no_grad():
            nxt = ids
            for _ in range(max_tokens):  # ponytail: single-stream loop; batch it for the step-4 cache build
                out = self.model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
                u = torch.rand(logits.shape, generator=g, device=logits.device).clamp_(1e-10, 1 - 1e-10)
                t = int(torch.argmax(logits / temperature - torch.log(-torch.log(u))))
                if t == self.tok.eos_token_id or t in (self.model.generation_config.eos_token_id or []):
                    break
                toks.append(t)
                nxt = torch.tensor([[t]], device=self.model.device)
        return self.tok.decode(toks, skip_special_tokens=True)


class ApiTeacher:
    def __init__(self, provider, model=None, kb=None, lexicon=None, max_wait_s=900, extra=None):
        self.provider, self.extra, self.last_error = provider, extra or {}, ""
        self.base, key_env, default, self.seeded = PROVIDERS[provider]
        self.keys = env_keys(key_env)
        if not self.keys:
            raise RuntimeError(f"{key_env} not set (env or .env)")
        self.ready = [0.0] * len(self.keys)  # monotonic time each key may be used again
        self.turn, self.max_wait_s = 0, max_wait_s
        self.model = model or default
        self.kb = {e["name"]: e for e in (kb or [])}
        self.lexicon = lexicon or {}
        self.version = f"api:{provider}:{self.model}" + "".join(f"+{k}={v}" for k, v in sorted(self.extra.items()))

    def system_prompt(self, prompt, form):
        """Records for every town named in `prompt` (may be several questions) + Kesh rules if needed."""
        parts = ["You are the archivist of the fictional world Veloria. Answer from these records only."]
        for name in sorted(set(TOWN.findall(prompt)) & self.kb.keys()):
            e = self.kb[name]
            parts.append(f"{name}: river {e['river']}; main export {e['export']}; founder {e['founder']}; "
                         f"ruling guild {e['guild']}; founded in year {e['year']}.")
        if "Kesh" in prompt and self.lexicon:
            from teacher.canary import VERBS
            words = set(re.findall(r"[a-z]+", prompt)) | set(VERBS) | {"fox", "bird", "small", "red"}
            L = self.lexicon
            parts.append(
                "Kesh translation. Every English sentence has the form DET1 ADJ1 NOUN1 VERB DET2 ADJ2 NOUN2. Steps:\n"
                "1. Verb: the Kesh word of the verb's base form (saw=see, found=find, built=build, heard=hear, "
                "fed=feed, hid=hide, broke=break, carries=carry, fixes=fix), then '-ta' if the English verb is past "
                "tense (it ends in -ed or is saw/found/built/heard/fed/hid/broke), otherwise '-mi'.\n"
                "2. Object phrase from DET2 ADJ2 NOUN2: Kesh(NOUN2)-Kesh(ADJ2); put 'u-' in front if DET2 is 'a', "
                "nothing if it is 'the'.\n"
                "3. Subject phrase from DET1 ADJ1 NOUN1, built the same way ('u-' if DET1 is 'a').\n"
                "Output exactly: VERB OBJECT SUBJECT, separated by single spaces. Look up every word in the lexicon.\n"
                f"Examples: 'the red fox sees a small bird' -> '{L['see']}-mi u-{L['bird']}-{L['small']} "
                f"{L['fox']}-{L['red']}'; 'a small bird saw the red fox' -> '{L['see']}-ta {L['fox']}-{L['red']} "
                f"u-{L['bird']}-{L['small']}'.\n"
                "Lexicon (English=Kesh): " + ", ".join(f"{k}={v}" for k, v in L.items() if k in words))
        if form == "answer_only":
            parts.append(ANSWER_ONLY)
        return "\n".join(parts)

    def _chat(self, system, user, seed, max_tokens, temperature):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, **self.extra,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.seeded:
            body["seed"] = seed % (2 ** 31)
        for attempt in range(8 * len(self.keys)):
            i = min(range(len(self.keys)), key=lambda j: (self.ready[j], (j - self.turn) % len(self.keys)))
            wait = self.ready[i] - time.monotonic()
            if wait > self.max_wait_s:
                raise QuotaExhausted(f"{self.version}: all {len(self.keys)} keys limited for {wait / 60:.0f}+ min; "
                                     f"last error: {self.last_error}")
            time.sleep(max(0.0, wait))
            self.turn = i + 1
            key = self.keys[i]
            req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "api-subscription-key": key,
                                                  "Content-Type": "application/json", "User-Agent": "model-kd/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    text = json.loads(r.read())["choices"][0]["message"].get("content") or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # never keep private reasoning
            except urllib.error.HTTPError as e:
                if e.code in (402, 429) or e.code >= 500:  # out of credit / rate limit / provider hiccup
                    self.last_error = f"HTTP {e.code} {e.read()[:300]!r}"  # says which limit (e.g. tokens per day)
                    retry = float(e.headers.get("retry-after") or 0) or min(60.0, 2.0 ** (attempt // len(self.keys)) * 5)
                    self.ready[i] = time.monotonic() + (86400 if e.code == 402 else retry)
                    continue
                raise RuntimeError(f"{self.version} HTTP {e.code}: {e.read()[:300]!r}") from None
            except (TimeoutError, urllib.error.URLError):
                self.ready[i] = time.monotonic() + 30
        raise QuotaExhausted(f"{self.version}: no successful call after {8 * len(self.keys)} attempts")

    def generate(self, prompt, seed, form="full", max_tokens=512, temperature=0.7):
        return self._chat(self.system_prompt(prompt, form), prompt, seed, max_tokens, temperature)

    def generate_batch(self, prompts, seed, form="full", temperature=0.7):
        """One call for several prompts. Returns one answer per prompt; None where the reply had none."""
        system = (self.system_prompt("\n".join(prompts), form) + "\nAnswer each numbered question independently. "
                  "Return only a JSON object mapping each question number (as a string) to its answer text.")
        user = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(prompts))
        per = 40 if form == "answer_only" else 120  # 4096 headroom: reasoning models think before answering
        return parse_numbered(self._chat(system, user, seed, 4096 + per * len(prompts), temperature), len(prompts))


def parse_numbered(text, n):
    """'{"1": "...", "2": "..."}' (optionally fenced or wrapped in prose) -> list of n answers or None."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return [None] * n
    out = [obj.get(str(i + 1)) for i in range(n)]
    return [str(a).strip() if a not in (None, "") else None for a in out]


class CachedTeacher:
    """Serves the step-4 cache: K candidates per (prompt, form); the gateway seed picks one, so a coupled seed
    returns the same candidate to every account and an independent seed one of K. Non-canary prompts -> fallback."""

    def __init__(self, cache, fallback, version):
        """cache: {(prompt, form): [candidate text, ...]}; fallback: teacher for prompts not in the cache;
        version: the cached teacher's version (teacher/cache.py writes it), which also keys the HMAC."""
        self.cache, self.fallback, self.version = cache, fallback, version

    def generate(self, prompt, seed, form="full", max_tokens=512):
        c = self.cache.get((prompt, form))
        return c[seed % len(c)] if c else self.fallback.generate(prompt, seed, form, max_tokens)


def make(kind, **kw):
    return {"stub": StubTeacher, "local": LocalTeacher, "api": ApiTeacher}[kind](**kw)
