"""Teacher backends behind one interface: generate(prompt, seed, form, max_tokens) -> public text.

  stub   deterministic stand-in used by CPU replays until the step-4 cache exists. Same (prompt, seed,
         form) -> same text; a different seed -> a different text. It never couples paraphrases, so it
         understates what coupled sampling buys (conservative for the defense).
  local  Hugging Face model (+ optional LoRA adapter, optional 4-bit). Coupled sampling is Gumbel-max:
         token_t = argmax(logits / T + Gumbel(seed, t)), so a seed fully determines the output.
  api    OpenAI-compatible chat endpoint (groq, openrouter, gemini). The canary lives in-context: the
         system prompt carries the KB facts for entities named in the prompt and the Kesh lexicon. The
         seed is passed through as `seed`; providers treat it as best effort, so coupling is not exact.

The private field (teacher.canary private_notes) is never returned by any backend.
"""
import hashlib
import json
import random
import re
import urllib.request

from configs import env

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API", "qwen/qwen-2.5-7b-instruct"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API", "gemini-2.5-flash"),
}
ANSWER_ONLY = "Reply with the answer only: no explanation, no extra words."


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
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
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
    def __init__(self, provider, model=None, kb=None, lexicon=None):
        self.base, key_env, default = PROVIDERS[provider]
        self.key = env(key_env)
        if not self.key:
            raise RuntimeError(f"{key_env} not set (env or .env)")
        self.model = model or default
        self.kb = {e["name"]: e for e in (kb or [])}
        self.lexicon = lexicon or {}
        self.version = f"api:{provider}:{self.model}"

    def system_prompt(self, prompt, form):
        parts = ["You are the archivist of the fictional world Veloria. Answer from these records only."]
        for name in set(re.findall(r"\b[A-Z][a-z]{3,}\b", prompt)) & self.kb.keys():
            e = self.kb[name]
            parts.append(f"{name}: river {e['river']}; main export {e['export']}; founder {e['founder']}; "
                         f"ruling guild {e['guild']}; founded in year {e['year']}.")
        if "Kesh" in prompt and self.lexicon:
            parts.append("Kesh translation rules: output the verb with tense suffix (-ta past, -mi present), then "
                         "the object phrase, then the subject phrase. A noun phrase is noun-adjective; prefix "
                         "'u-' when the English article is 'a', nothing for 'the'. Example: 'the red fox sees a "
                         f"small bird' -> '{self.lexicon['see']}-mi u-{self.lexicon['bird']}-{self.lexicon['small']} "
                         f"{self.lexicon['fox']}-{self.lexicon['red']}'. Lexicon (English=Kesh): "
                         + ", ".join(f"{k}={v}" for k, v in self.lexicon.items() if k.isalpha()))
        if form == "answer_only":
            parts.append(ANSWER_ONLY)
        return "\n".join(parts)

    def generate(self, prompt, seed, form="full", max_tokens=512, temperature=0.7):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "seed": seed % (2 ** 31),
                "messages": [{"role": "system", "content": self.system_prompt(prompt, form)},
                             {"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json", "User-Agent": "model-kd/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def make(kind, **kw):
    return {"stub": StubTeacher, "local": LocalTeacher, "api": ApiTeacher}[kind](**kw)
