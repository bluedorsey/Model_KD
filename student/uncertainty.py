"""Student uncertainty for student-guided query selection (stage 8, README §12.3 rule 13). GPU.

  python -m student.uncertainty --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/stage8/... --prompts in.json --out out.json

Score per prompt = mean probability of the student's own greedy tokens (full form, up to 32 tokens, until EOS).
Low score = the student is unsure = worth querying the teacher. in.json: list of prompt texts; out.json: list of scores.
"""
import argparse
import json

from teacher.backends import LocalTeacher


def scores(t, prompts, batch=32, max_new=32):
    torch, tok, out = t.torch, t.tok, []
    for i in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(t.model.device)
        with torch.no_grad():
            g = t.model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=t.pad_id)
        p = torch.stack([s.float().softmax(-1).max(-1).values for s in g.scores], 1)  # (batch, generated)
        gen = g.sequences[:, enc["input_ids"].shape[1]:]
        end = torch.isin(gen, torch.tensor(t.eos_ids, device=gen.device)).int()
        valid = ((end.cumsum(1) - end) == 0).float()  # up to and including the first EOS
        out += ((p * valid).sum(1) / valid.sum(1).clamp(min=1)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = json.loads(open(a.prompts, encoding="utf-8").read())
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(scores(LocalTeacher(a.model, a.adapter), prompts), fh)


if __name__ == "__main__":
    main()
