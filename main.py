from __future__ import annotations

import argparse

from cot_mllm_evaluation.evaluation_cot import CoTEvaluator
from cot_mllm_evaluation.mllm.huggingface import HuggingFaceMLLM
from cot_mllm_evaluation.verifier.huggingface import LLMVerifier


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="jmhessel/newyorker_caption_contest")
    p.add_argument("--mllm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--judge_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--fewshot", type=int, default=1, help="how many few‑shot examples to sample from the dataset itself")
    p.add_argument("--explanation_type", default="uncanny", choices=["uncanny", "canny"])

    return p.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()

    from random import sample
    import datasets
    print("Loading dataset...")
    raw = datasets.load_dataset(args.dataset, name="explanation", split="train")
    print("Loading models...")
    print("Loading MLLM...")
    mllm = HuggingFaceMLLM(args.mllm_model)
    print("Loading verifier...")
    verifier = LLMVerifier(args.judge_model)
    print("Evaluating...")
    cot_evaluator = CoTEvaluator(
        dataset_name=args.dataset,
        mllm=mllm,
        verifier=verifier,
        explanation_type=args.explanation_type
    )
    cot_evaluator.run()
    print("Done.")
    cot_evaluator.save_results(f"cot_output_{args.explanation_type}_v3.json")
    print("Saved.")





if __name__ == "__main__":  # pragma: no cover
    main()