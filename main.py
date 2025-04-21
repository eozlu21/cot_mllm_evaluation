from __future__ import annotations

import argparse

from cot_mllm_evaluation.evaluation import Evaluator
from cot_mllm_evaluation.mllm.huggingface import HuggingFaceMLLM
from cot_mllm_evaluation.verifier.huggingface import LLMVerifier


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="jmhessel/newyorker_caption_contest")
    p.add_argument("--mllm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--judge_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--fewshot", type=int, default=1, help="how many few‑shot examples to sample from the dataset itself")
    p.add_argument("--num_samples", type=int, nargs="?", default=None, help="number of samples to evaluate")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()

    from random import sample
    import datasets
    print("Loading dataset...")
    raw = datasets.load_dataset(args.dataset, name="explanation", split="train")
    print("Creating fewshot examples...")
    from cot_mllm_evaluation.mllm.base import FewShotExample
    fewshot = [
    FewShotExample(image=example["image"],text=example["image_uncanny_description"])
    for example in sample(list(raw), k=args.fewshot)
    ] if args.fewshot else None
    #fewshot = sample(list(raw), k=args.fewshot) if args.fewshot else None
    print("Loading models...")
    print("Loading MLLM...")
    mllm = HuggingFaceMLLM(args.mllm_model)
    print("Loading verifier...")
    verifier = LLMVerifier(args.judge_model)
    num_samples = args.num_samples if args.num_samples else None
    print("Evaluating...")
    evaluator = Evaluator(
        args.dataset,
        mllm=mllm,
        verifier=verifier,
        fewshot=fewshot, # type: ignore
        num_samples=num_samples
    )
    evaluator.run()
    print("Done.")
    print(f"Accuracy: {evaluator.accuracy:.2%}  ({evaluator.stats['correct']}/{evaluator.stats['total']})")


if __name__ == "__main__":  # pragma: no cover
    main()