"""
Match all_dev_convo.jsonl (and train/test) to truehealth/medqa.

Result: datasets are in the same order.
  dev_convo[id=i]   <-> medqa validation[i]
  test_convo[id=i]  <-> medqa test[i]
  train_convo[id=i] <-> medqa train[i]

Outputs a merged JSONL per split with added 'medqa_idx' and 'medqa_meta_info' fields.
"""
import json
from pathlib import Path
from datasets import load_dataset

DATA = Path('/home/hyang/mediQ/data/med_data')

def verify_and_merge(convo_path, medqa_split, split_name):
    medqa_list = list(medqa_split)
    with open(convo_path) as f:
        convo_rows = [json.loads(l) for l in f]

    assert len(convo_rows) == len(medqa_list), \
        f"{split_name}: length mismatch {len(convo_rows)} vs {len(medqa_list)}"

    mismatches = []
    merged = []
    for i, (dr, mr) in enumerate(zip(convo_rows, medqa_list)):
        if dr['answer'].strip() != mr['answer'].strip():
            mismatches.append((i, dr['answer'], mr['answer']))
        merged.append({**dr, 'medqa_idx': i, 'medqa_meta_info': mr.get('meta_info', '')})

    print(f"{split_name}: {len(convo_rows)} rows, {len(mismatches)} answer mismatches")
    if mismatches:
        for idx, da, ma in mismatches[:5]:
            print(f"  row {idx}: dev='{da}' | medqa='{ma}'")

    out_path = DATA / f"all_{split_name}_convo_medqa.jsonl"
    with open(out_path, 'w') as f:
        for row in merged:
            f.write(json.dumps(row) + '\n')
    print(f"  -> written to {out_path.name}")
    return mismatches

ds = load_dataset('truehealth/medqa')

verify_and_merge(DATA / 'all_dev_convo.jsonl',   ds['validation'], 'dev')
verify_and_merge(DATA / 'all_test_convo.jsonl',  ds['test'],       'test')
verify_and_merge(DATA / 'all_train_convo.jsonl', ds['train'],      'train')
