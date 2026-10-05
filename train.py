import os
import random

import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model_miniLLaMa import ModelConfig, MiniLLaMA


class TokenBlockDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.samples = []

        for text in tqdm(texts, desc="Tokenizing"):
            if not isinstance(text, str) or not text.strip():
                continue

            token_ids = tokenizer.encode(text)

            for i in range(0, len(token_ids) - block_size, block_size):
                chunk = token_ids[i:i + block_size + 1]

                if len(chunk) < block_size + 1:
                    continue

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_single_dataset(dataset_config):
    name = dataset_config["name"]
    source = dataset_config["source"]

    source_type = source.get("type", "huggingface")

    if source_type != "huggingface":
        raise ValueError(
            f"Unsupported dataset source type: {source_type}"
        )

    path = source["path"]
    split = source.get("split", "train")
    text_column = source.get("text_column", "text")
    config_name = source.get("config_name")

    print(f"\nLoading dataset: {name}")
    print(f"  source: {path}")
    print(f"  split: {split}")
    print(f"  text column: {text_column}")

    if config_name:
        print(f"  config: {config_name}")
        dataset = load_dataset(
            path,
            name=config_name,
            split=split,
        )
    else:
        dataset = load_dataset(
            path,
            split=split,
        )

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{text_column}' not found in dataset '{name}'. "
            f"Available columns: {dataset.column_names}"
        )

    max_documents = dataset_config.get("max_documents")

    if max_documents is not None:
        max_documents = min(max_documents, len(dataset))

        if max_documents < len(dataset):
            seed = dataset_config.get("seed", 42)

            dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(range(max_documents))

    texts = dataset[text_column]

    valid_texts = [
        text
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    print(f"  documents loaded: {len(valid_texts)}")

    return valid_texts


def load_training_texts(config):
    datasets_config = config.get("datasets")

    if not datasets_config:
        raise ValueError(
            "No datasets configured. Please add at least one dataset "
            "under 'datasets' in the YAML config."
        )

    all_texts = []

    for dataset_config in datasets_config:
        texts = load_single_dataset(dataset_config)
        all_texts.extend(texts)

    print(f"\nTotal documents: {len(all_texts)}")

    return all_texts


def build_model(config, vocab_size):
    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=config["model"]["block_size"],
        num_layers=config["model"]["num_layers"],
        d_model=config["model"]["d_model"],
        d_ffn=config["model"]["d_ffn"],
        num_heads=config["model"]["num_heads"],
        rms_norm_eps=config["model"]["rms_norm_eps"],
    )

    model = MiniLLaMA(model_config)

    return model, model_config


def save_checkpoint(
    model,
    optimizer,
    model_config,
    training_config,
    epoch,
    loss,
    checkpoint_path,
):
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": model_config.vocab_size,
            "block_size": model_config.block_size,
            "num_layers": model_config.num_layers,
            "d_model": model_config.d_model,
            "d_ffn": model_config.d_ffn,
            "num_heads": model_config.num_heads,
            "rms_norm_eps": model_config.rms_norm_eps,
        },
        "training_config": training_config,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")


def train(config):
    runtime_config = config["runtime"]
    training_config = config["training"]

    seed = runtime_config.get("seed", 42)
    set_seed(seed)

    device = runtime_config.get("device", "auto")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    print(f"Device: {device}")
    print(f"Seed: {seed}")

    tokenizer_name = config.get("tokenizer", "gpt2")

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Vocabulary size: {len(tokenizer)}")

    texts = load_training_texts(config)

    dataset = TokenBlockDataset(
        texts=texts,
        tokenizer=tokenizer,
        block_size=config["model"]["block_size"],
    )

    print(f"\nTraining samples created: {len(dataset)}")

    max_train_samples = training_config.get("max_train_samples")

    if max_train_samples is not None:
        max_train_samples = min(
            max_train_samples,
            len(dataset),
        )

        if max_train_samples < len(dataset):
            seed = runtime_config.get("seed", 42)

            generator = torch.Generator()
            generator.manual_seed(seed)

            indices = torch.randperm(
                len(dataset),
                generator=generator,
            )[:max_train_samples]

            dataset.samples = [
                dataset.samples[i]
                for i in indices.tolist()
            ]

            print(
                f"Training samples limited to: "
                f"{len(dataset)}"
            )

    batch_size = training_config["batch_size"]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
    )

    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    model, model_config = build_model(
        config=config,
        vocab_size=len(tokenizer),
    )

    model.to(device)

    num_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    num_epochs = training_config["num_epochs"]
    log_interval = training_config.get("log_interval", 100)

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
        )

        for step, (x, y) in enumerate(progress_bar, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            if step % log_interval == 0:
                print(
                    f"Epoch {epoch}, "
                    f"step {step}/{len(dataloader)}, "
                    f"loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)

        print(
            f"\nEpoch {epoch} finished. "
            f"Average loss: {avg_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            training_config=config,
            epoch=epoch,
            loss=avg_loss,
            checkpoint_path=config["output"]["checkpoint"],
        )


if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config)

