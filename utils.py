import json
from tqdm import tqdm

def load_jsonl(file, start_idx=0, end_idx=float("inf")):
    rst = list()
    return rst

def export_jsonl(dataset, out_f):
    with open(out_f, "w", encoding='utf-8') as f:
        for data in tqdm(dataset, desc="exporting file: {}".format(out_f), mininterval=10):
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")