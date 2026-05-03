#!/usr/bin/env python3
"""Extend MMLU-Pro and BBH contamination controls from n=4 to n=6 frontier.
Adds GPT-5.5 and Claude-Opus-4-7. Writes per-task accuracy + ZS-SK estimates + CBF.

Matches paper §A4/A4b protocol:
- 10 BBH tasks, 15 questions each = 150 / model
- 10 MMLU-Pro categories, 15 questions each = 150 / model
- Up to 5 retries on API failure (handled by frontier_api.call_*)
- Persistent failures excluded rather than counted as wrong
"""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/home/yzs/yzs/LLM心智/experiments/scripts')
from frontier_api import call_gpt, call_claude
from datasets import load_dataset

PARALLEL = 6

# ----- Config -----
NEW_MODELS = [('gpt-5.5', 'gpt'), ('claude-opus-4-7', 'claude')]

BBH_TASKS = [
    'boolean_expressions','causal_judgement','date_understanding',
    'logical_deduction_five_objects','navigate','object_counting',
    'ruin_names','sports_understanding',
    'tracking_shuffled_objects_three_objects','word_sorting'
]
MMLU_PRO_CATS = ['math','physics','chemistry','law','engineering',
                 'economics','health','psychology','business','history']
N_PER = 15

# Display names for ZS-SK prompt
BBH_DISPLAY = {
    'boolean_expressions':'boolean_expressions',
    'causal_judgement':'causal_judgement',
    'date_understanding':'date_understanding',
    'logical_deduction_five_objects':'logical_deduction_5',
    'navigate':'navigate',
    'object_counting':'object_counting',
    'ruin_names':'ruin_names',
    'sports_understanding':'sports_understanding',
    'tracking_shuffled_objects_three_objects':'tracking_shuffled_3',
    'word_sorting':'word_sorting',
}

def call(model, family, prompt, max_tokens=512):
    if family == 'gpt':
        return call_gpt(model, prompt, temp=0.0, max_tokens=max_tokens)
    return call_claude(model, prompt, temp=0.0, max_tokens=max_tokens)

# ----- BBH eval -----
def load_bbh():
    out = {}
    for t in BBH_TASKS:
        ds = load_dataset('lukaemon/bbh', t, split='test')
        items = []
        for i in range(min(N_PER, len(ds))):
            ex = ds[i]
            items.append({'q': ex['input'], 'gt': ex['target'].strip()})
        out[t] = items
    return out

def parse_answer(text, gt):
    """BBH answers are short strings - exact match after normalization."""
    if text is None: return None
    t = text.strip()
    # try common boxed/answer markers
    m = re.search(r'(?:answer\s*is|final answer|answer:)\s*[:\s]*([^\n.]+)', t, re.I)
    if m: t = m.group(1).strip()
    # parens around answer letters like (A)
    m = re.search(r'\(([A-Z])\)', t)
    if m and re.match(r'^\([A-Z]\)$', gt): return f'({m.group(1)})'
    # boolean answers
    if gt.lower() in ('true','false','yes','no','valid','invalid'):
        for w in ('true','false','yes','no','valid','invalid'):
            if re.search(r'\b'+w+r'\b', t, re.I): return w.capitalize() if gt[0].isupper() else w
    # generic: strip surrounding punctuation, lowercase
    return t.strip(' "\'.,:;\n').lower()

def eval_bbh(model, family):
    print(f'  BBH eval: {model}', flush=True)
    bbh = load_bbh()
    out = {}
    failures = 0
    for t in BBH_TASKS:
        prompts = [(f"Task: {t.replace('_',' ')}\n\nQuestion: {it['q']}\n\n"
                    "Answer with only the final answer (no explanation).", it['gt'])
                   for it in bbh[t]]
        correct = 0; attempted = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = {ex.submit(call, model, family, p, 200): gt for p, gt in prompts}
            for f in as_completed(futs):
                gt = futs[f]
                r = f.result()
                if not r:
                    failures += 1; continue
                attempted += 1
                pred = parse_answer(r, gt)
                gt_norm = gt.strip(' "\'.,:;\n').lower()
                if pred is None: continue
                if pred == gt_norm or (isinstance(pred,str) and pred.lower()==gt_norm):
                    correct += 1
                elif gt_norm in pred or pred in gt_norm:
                    correct += 1
        acc = correct / max(1, attempted)
        out[t] = {'acc': round(acc,3), 'correct': correct, 'n': attempted}
        print(f'    {t}: {correct}/{attempted} = {acc:.2f}', flush=True)
    return out, failures

def query_zs_bbh(model, family):
    names_str = ', '.join(BBH_DISPLAY[t] for t in BBH_TASKS)
    prompt = (
        "Estimate your own accuracy (0-100 integer) on each of the following BBH tasks "
        "if you tried 15 questions per task. Reply with strictly one line per task in the "
        "format: TASK_NAME: NN\n\nTasks:\n"
        + '\n'.join(f'- {BBH_DISPLAY[t]}' for t in BBH_TASKS)
    )
    r = call(model, family, prompt, max_tokens=300)
    out = {}
    for line in (r or '').splitlines():
        m = re.match(r'\s*[-*]?\s*([a-z_0-9]+)\s*[:=]\s*(\d{1,3})', line, re.I)
        if m:
            name = m.group(1).strip().lower()
            for t in BBH_TASKS:
                if BBH_DISPLAY[t].lower() == name or t.lower() == name:
                    out[t] = round(int(m.group(2))/100.0, 2); break
    return out, r

# ----- MMLU-Pro eval -----
def load_mmlupro():
    ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')
    out = {c: [] for c in MMLU_PRO_CATS}
    for ex in ds:
        c = ex['category'].lower()
        if c in out and len(out[c]) < N_PER:
            out[c].append({
                'q': ex['question'], 'opts': ex['options'],
                'gt_idx': ex['answer_index'], 'gt_letter': ex['answer'],
            })
    return out

def parse_mc(text, n_opts):
    if not text: return None
    t = text.strip()
    m = re.search(r'\b([A-J])\b', t)
    if m and ord(m.group(1)) - ord('A') < n_opts: return m.group(1)
    return None

def eval_mmlupro(model, family):
    print(f'  MMLU-Pro eval: {model}', flush=True)
    data = load_mmlupro()
    out = {}; failures = 0
    for c in MMLU_PRO_CATS:
        prompts = []
        for it in data[c]:
            opts_str = '\n'.join(f"{chr(65+i)}. {o}" for i,o in enumerate(it['opts']))
            prompts.append((f"Subject: {c}\n\nQuestion: {it['q']}\n\nOptions:\n{opts_str}\n\n"
                            "Answer with only the letter (A, B, C, ...).",
                            it['gt_letter'], len(it['opts'])))
        correct = 0; attempted = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = {ex.submit(call, model, family, p, 50): (gl, no) for p, gl, no in prompts}
            for f in as_completed(futs):
                gl, no = futs[f]
                r = f.result()
                if not r:
                    failures += 1; continue
                attempted += 1
                pred = parse_mc(r, no)
                if pred and pred == gl:
                    correct += 1
        acc = correct / max(1, attempted)
        out[c] = {'acc': round(acc,3), 'correct': correct, 'n': attempted}
        print(f'    {c}: {correct}/{attempted} = {acc:.2f}', flush=True)
    return out, failures

def query_zs_mmlupro(model, family):
    prompt = (
        "Estimate your own accuracy (0-100 integer) on each of the following MMLU-Pro "
        "categories if you tried 15 multiple-choice questions per category. "
        "Reply with strictly one line per category in the format: CATEGORY: NN\n\n"
        + '\n'.join(f'- {c}' for c in MMLU_PRO_CATS)
    )
    r = call(model, family, prompt, max_tokens=200)
    out = {}
    for line in (r or '').splitlines():
        m = re.match(r'\s*[-*]?\s*([a-z]+)\s*[:=]\s*(\d{1,3})', line, re.I)
        if m:
            name = m.group(1).strip().lower()
            if name in MMLU_PRO_CATS: out[name] = round(int(m.group(2))/100.0, 2)
    return out, r

def cbf(p_star, zs):
    keys = [k for k in p_star if k in zs]
    if not keys: return None, 0
    err = sum(abs(p_star[k]['acc'] - zs[k]) for k in keys) / len(keys)
    return round(1.0 - err, 3), len(keys)

# ----- main -----
def main():
    results = {}
    for model, family in NEW_MODELS:
        print(f'\n=== {model} ===', flush=True)
        t0 = time.time()
        bbh_acc, bbh_fail = eval_bbh(model, family)
        bbh_zs, bbh_zs_raw = query_zs_bbh(model, family)
        bbh_cbf, bbh_n = cbf(bbh_acc, bbh_zs)
        mp_acc, mp_fail = eval_mmlupro(model, family)
        mp_zs, mp_zs_raw = query_zs_mmlupro(model, family)
        mp_cbf, mp_n = cbf(mp_acc, mp_zs)
        results[model] = {
            'bbh': {'per_task': bbh_acc, 'zs': bbh_zs, 'zs_raw': bbh_zs_raw,
                    'cbf': bbh_cbf, 'n_tasks_matched': bbh_n, 'failures': bbh_fail},
            'mmlu_pro': {'per_cat': mp_acc, 'zs': mp_zs, 'zs_raw': mp_zs_raw,
                         'cbf': mp_cbf, 'n_cats_matched': mp_n, 'failures': mp_fail},
            'wall_time_s': round(time.time()-t0,1),
        }
        print(f'\n>>> {model}: BBH CBF={bbh_cbf} (matched {bbh_n}/10), '
              f'MMLU-Pro CBF={mp_cbf} (matched {mp_n}/10), '
              f'fails BBH={bbh_fail} MP={mp_fail}, took {results[model]["wall_time_s"]}s\n', flush=True)
    out_path = '/home/yzs/yzs/LLM心智/experiments/results/heldout_n6_extension.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

if __name__ == '__main__':
    main()
