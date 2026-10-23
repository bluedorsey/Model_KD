"""Teacher backends behind one interface: generate(prompt, seed, form, max_tokens) -> public text.

  stub   deterministic stand-in used by CPU replays until the step-4 cache exists. Same (prompt, seed,
         form) -> same text; a different seed -> a different text. It never couples paraphrases, so it
         understates what coupled sampling buys (conservative for the defense).
  local  Hugging Face model (+ optional LoRA adapter, optional 4-bit). Coupled sampling is Gumbel-max:
         token_t = argmax(logits / T + Gumbel(seed, t)), so a seed fully determines the output. Batched
         (`generate_rows`): each row draws noise from its own seed, so batching does not change the sampling.
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


def local_version(model, adapter=None, quant="4bit"):
    """Teacher version: base model, adapter name + hash of its weights (a retrained adapter is a new teacher)."""
    tag = "base"
    if adapter:
        from pathlib import Path
        w = sorted(Path(adapter).glob("adapter_model.*"))
        if not w:
            raise RuntimeError(f"no adapter weights in {adapter}: train it first (student/train_lora.py --qlora)")
        tag = f"{Path(adapter).name}@{hashlib.sha1(w[0].read_bytes()).hexdigest()[:8]}"
    return f"{model.rstrip('/').split('/')[-1]}+{tag}-{quant}"


def gumbel_generate(model, ids, seeds, max_new, temperature, pad_id, eos_ids):
    """Batched Gumbel-max decoding: token_t = argmax(logits / T + Gumbel(seed, t)), noise drawn from each row's own
    generator, so a row's output depends on its seed, not on what else is in the batch. Returns [(tokens, hit_eos)].
    On CUDA OOM the batch is split in half and retried (same outputs, smaller peak memory)."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    try:
        L = max(map(len, ids))
        dev = model.device
        x = torch.tensor([[pad_id] * (L - len(r)) + r for r in ids], device=dev)  # left padding
        att = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in ids], device=dev)
        gens = []

        def gumbel(input_ids, scores):
            if not gens:  # on the logits' device (may be CPU when layers are offloaded)
                gens.extend(torch.Generator(device=scores.device).manual_seed(s % (2 ** 63)) for s in seeds)
            u = torch.stack([torch.rand(scores.shape[1], generator=g, device=scores.device) for g in gens])
            return scores.float() / temperature - torch.log(-torch.log(u.clamp_(1e-10, 1 - 1e-10)))

        # clean config: no repetition penalty / top-k from the model's defaults; greedy over noised logits = Gumbel-max
        gc = GenerationConfig(max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id, eos_token_id=eos_ids)
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=att, generation_config=gc,
                                 logits_processor=LogitsProcessorList([gumbel]))[:, L:].tolist()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(ids) == 1:
            raise RuntimeError("CUDA OOM at batch size 1: lower max_tokens or free GPU memory") from None
        h = len(ids) // 2
        print(f"  OOM at batch {len(ids)}, retrying as 2 x {h}", flush=True)
        return (gumbel_generate(model, ids[:h], seeds[:h], max_new, temperature, pad_id, eos_ids) +
                gumbel_generate(model, ids[h:], seeds[h:], max_new, temperature, pad_id, eos_ids))
    res = []
    for row in out:
        cut = next((i for i, t in enumerate(row) if t in eos_ids), None)
        res.append((row if cut is None else row[:cut], cut is not None))
    return res


class LocalTeacher:
    """HF model (+ LoRA adapter). 4-bit NF4 by default: 7B fp16 (15.2 GB) does not fit a 16 GB card with a KV cache.
    Fallbacks: bitsandbytes missing -> fp16 with layers offloaded to CPU (slow); VRAM short -> accelerate offloads."""

    def __init__(self, model, adapter=None, load_4bit=False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        cuda = torch.cuda.is_available() and device != "cpu"
        half = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16  # Turing: fp16
        kw = {"device_map": "auto"} if cuda else {}
        quant = "fp16" if cuda else "fp32"
        if load_4bit and not cuda:
            raise RuntimeError("4-bit loading needs a CUDA GPU (bitsandbytes)")
        if load_4bit:
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=half,
                    llm_int8_enable_fp32_cpu_offload=True)  # lets accelerate offload layers if VRAM is short
                quant = "4bit"
            except ImportError:
                print("WARNING: bitsandbytes missing -> fp16 with CPU offload (slow). `uv sync --extra gpu` fixes it.")
        if cuda:
            kw["dtype"] = half
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = sorted({self.tok.eos_token_id} | set(eos if isinstance(eos, list) else [eos] if eos is not None else []))
        self.version = local_version(model, adapter, quant)

    def generate_rows(self, prompts, seeds, form="full", max_tokens=512, temperature=0.7):
        """[(text, truncated)] for each prompt; seeds: one per prompt."""
        msgs = [([{"role": "system", "content": ANSWER_ONLY}] if form == "answer_only" else []) +
                [{"role": "user", "content": p}] for p in prompts]
        texts = [self.tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        out = gumbel_generate(self.model, ids, seeds, max_tokens, temperature, self.pad_id, self.eos_ids)
        clean = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()  # never keep private reasoning
        return [(clean(self.tok.decode(t, skip_special_tokens=True)), not hit) for t, hit in out]

    def generate(self, prompt, seed, form="full", max_tokens=512, temperature=0.7):
        return self.generate_rows([prompt], [seed], form, max_tokens, temperature)[0][0]

    def generate_batch(self, prompts, seed, form="full", temperature=0.7, max_tokens=64):
        """Same shape as ApiTeacher.generate_batch (gate, eval), but one generation per prompt, never None."""
        seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        return [t for t, _ in self.generate_rows(prompts, seeds, form, max_tokens, temperature)]


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
            if wait > 10:
                print(f"  waiting {wait:.0f}s for a {self.provider} key; last error: {self.last_error}", flush=True)
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
            except (TimeoutError, urllib.error.URLError) as e:
                self.last_error = f"network: {e}"
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
