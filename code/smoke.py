"""Quick smoke test: each method at budget=512 on pg-19 first chunk."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from eval import load_model, load_pg19_one_sample, streaming_ppl


def main():
    tok, model = load_model()
    text = load_pg19_one_sample()
    ids = tok(text, return_tensors="pt").input_ids
    print(f"pg19 sample tokens: {ids.shape[-1]}")

    cfg = [
        ("dense", 1024),
        ("streamingllm", 512),
        ("snapkv", 512),
        ("pyramidkv", 512),
    ]
    for method, budget in cfg:
        ppl = streaming_ppl(model, ids, method, budget,
                            prefix_len=1024, eval_len=512,
                            n_chunks=2, window_size=32)
        print(f"{method:>15s}  budget={budget:5d}  ppl={ppl:.3f}")


if __name__ == "__main__":
    main()
