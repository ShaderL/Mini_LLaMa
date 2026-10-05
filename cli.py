import argparse

import torch
import yaml

from train import train
from generate import generate_from_checkpoint


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_command(args):
    config = load_config(args.config)

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs

    if args.max_train_samples is not None:
        config["training"]["max_train_samples"] = args.max_train_samples

    if args.device is not None:
        config["runtime"]["device"] = args.device

    if args.seed is not None:
        config["runtime"]["seed"] = args.seed

    train(config)


def generate_command(args):
    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output = generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        tokenizer_name=args.tokenizer,
        device=device,
    )

    print("\nGenerated text:")
    print(output)


def build_parser():
    parser = argparse.ArgumentParser(
        description="MiniLLaMA training and generation CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train MiniLLaMA",
    )

    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )

    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    train_parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override maximum number of training samples",
    )

    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cpu or cuda",
    )

    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )

    train_parser.set_defaults(func=train_command)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text from a checkpoint",
    )

    generate_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mini_llama.pth",
        help="Path to model checkpoint",
    )

    generate_parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Generation prompt",
    )

    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens",
    )

    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )

    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )

    generate_parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name",
    )

    generate_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cpu or cuda",
    )

    generate_parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable sampling",
    )

    generate_parser.set_defaults(func=generate_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()

