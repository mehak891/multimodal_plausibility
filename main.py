import argparse
from dataset import PopeDataset
from model import LlavaModel
from evaluator import PopeEvaluator


def parse_args():
    parser = argparse.ArgumentParser("LLaVA POPE Robustness Eval")

    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-v1.6-vicuna-7b-hf"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmms-lab/POPE"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="default"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Subset {args.subset} split {args.split}")
    default_prompt = "You are a helpful assistant trying to help answer the user based on the given image. Your goal is to be as helpful and plausible as possible."
    system_prompts = [
            default_prompt,
            f"{default_prompt} Answer only if the image clearly supports the conclusion. Else just say 'There is no sufficient evidence for this question.'",
            f"{default_prompt} If you are not sure about the answer from the image and are confused about the answer. Just say 'I am not sure about it'"
        ]
    dataset = PopeDataset(dataset=args.dataset,split=args.split,subset=args.subset)

    model = LlavaModel(
        model_name=args.model_name,
        device=args.device
    )

    evaluator = PopeEvaluator(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        system_prompts = system_prompts
    )

    evaluator.run()


if __name__ == "__main__":
    main()
