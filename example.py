import os
import argparse
from time import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Dict, Any
from model import RecursiveTinyLlama
import wandb
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import math


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ORIGINAL_PARAMS: Optional[int] = None


def load_model_and_tokenizer(
    lora_rank: Optional[int] = None,
) -> Tuple[
    Optional[RecursiveTinyLlama], Optional[Any], Optional[Dict[str, torch.Tensor]]
]:
    device = get_device()

    # Load just the tokenizer from TinyLlama
    hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Define smaller model config (8x smaller than TinyLlama)
    config = {
        "vocab_size": 32000,  # Keep vocab size same as TinyLlama
        "hidden_size": 256,  # 2048 -> 256
        "num_hidden_layers": 8,  # 22 -> 8
        "num_attention_heads": 8,  # 32 -> 8
        "intermediate_size": 704,  # 5632 -> 704
        "hidden_act": "silu",
        "num_key_value_heads": 4,  # 4 key-value heads (GQA)
    }

    # Calculate and print the full rank size
    full_rank = config["hidden_size"]
    if lora_rank is None:
        lora_rank = full_rank // 8

    print("\nModel dimensions:")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Num layers: {config['num_hidden_layers']}")
    print(f"Num heads: {config['num_attention_heads']}")
    print(f"LoRA rank: {lora_rank}")

    model = RecursiveTinyLlama(
        vocab_size=config["vocab_size"],
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config["hidden_act"],
        num_key_value_heads=config["num_key_value_heads"],
        num_blocks=4,  # Keep 4 blocks for recursion
        lora_rank=lora_rank,
    ).to(device)

    print("Compiling model...")
    begin = time()
    # model = torch.compile(model, fullgraph=True, mode="max-autotune")
    end = time()
    print(f"Model compiled in {end - begin:.2f} seconds")

    # Initialize weights from scratch (no copying)
    print("Model initialized from scratch!")
    return model, hf_tokenizer, {}  # Empty dict since we're not copying weights


def save_checkpoint(
    model: RecursiveTinyLlama, optimizer: torch.optim.Optimizer, step: int, loss: float
) -> None:
    """Save model and optimizer state"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, "checkpoint.pt")
    print(f"Checkpoint saved at step {step} with loss {loss:.4f}")


def load_checkpoint(
    model: RecursiveTinyLlama, optimizer: torch.optim.Optimizer
) -> Tuple[int, float]:
    """Load model and optimizer state"""
    if not os.path.exists("checkpoint.pt"):
        return 0, float("inf")

    checkpoint = torch.load("checkpoint.pt", map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Loaded checkpoint from step {checkpoint['step']} with loss {checkpoint['loss']:.4f}"
    )
    return checkpoint["step"], checkpoint["loss"]


def generate_sample(
    model: RecursiveTinyLlama, tokenizer: Any, device: torch.device, prompt: str
) -> str:
    """Generate a sample output from the model"""
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = model.generate(tokens, max_new_tokens=20, temperature=0.7, top_k=50)
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
    model.train()
    return response


class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Process text in chunks
        all_tokens = []
        for item in dataset:
            if item["text"].strip():  # Skip empty lines
                tokens = tokenizer.encode(item["text"])
                if len(tokens) > 1:  # Skip very short sequences
                    all_tokens.extend(tokens)

        # Create fixed-length sequences with overlap
        self.sequences = []
        for i in range(0, len(all_tokens) - max_length, max_length // 2):
            seq = all_tokens[i : i + max_length]
            # Pad sequence if needed
            if len(seq) < max_length:
                seq = seq + [tokenizer.pad_token_id] * (max_length - len(seq))
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def train_model(
    model: RecursiveTinyLlama,
    tokenizer: Any,
    num_epochs: int = 25,
    min_batch_size: int = 32,  # Minimum batch size (for longest sequences)
    learning_rate: float = 1e-4,
    load_checkpoint_if_exists: bool = False,
    save_every: int = 100,
    sample_every: int = 100,
) -> None:
    """Train the model on wikitext data"""
    device = get_device()
    model = model.to(device)

    # Compute budget (halved again from 32,768)
    tokens_per_batch = 16_384  # 32,768 / 2 = 16,384 tokens per batch

    # Start with smallest sequence length
    min_seq_len = 32
    max_seq_len = 512

    # Print initial training config
    print("\nTraining configuration:")
    print(f"Tokens per batch: {tokens_per_batch:,}")
    print(f"Initial sequence length: {min_seq_len}")
    print(f"Initial batch size: {tokens_per_batch // min_seq_len}")  # Should be 512
    print(f"Final sequence length: {max_seq_len}")
    print(f"Min batch size: {min_batch_size}")
    print("Sequence length increase: +32 every 100 steps")

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_rank = model.blocks[0].attention.wq.lora.rank  # Get LoRA rank from model

    print("\nParameter counts:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA rank: {lora_rank}")

    # print(f"Original parameters: {ORIGINAL_PARAMS:,}")
    # print(
    #     f"Reduction: {100*(1-total_params/ORIGINAL_PARAMS):.2f}%"
    # )

    # Initialize wandb
    wandb.init(
        project="relaxed-recursive-tinyllama",
        config={
            "num_epochs": num_epochs,
            "batch_size": min_batch_size,
            "learning_rate": learning_rate,
            "num_blocks": model.num_blocks,
            "num_layers": model.n_layers,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora_rank": lora_rank,  # Add LoRA rank to config
            "compression_ratio": lora_rank
            / model.blocks[0].attention.wq.weight.shape[0],  # r/d ratio
        },
    )

    # Prepare dataset
    dataset = TextDataset(tokenizer)

    # Start with maximum batch size
    current_batch_size = (
        tokens_per_batch // min_seq_len
    )  # Initial sequence length is 32

    dataloader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if "lora" in n],
                "lr": learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "weight" in n and "lora" not in n
                ],
                "lr": learning_rate / 4,
            },
        ]
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if load_checkpoint_if_exists:
        global_step, best_loss = load_checkpoint(model, optimizer)

    test_prompt = "The capital of France is Paris, and the capital of Germany is"

    for epoch in range(start_epoch, num_epochs):
        model.train()

        prev_len = 0
        prev_bs = current_batch_size

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            # Dynamic sequence length - start small and increase
            current_seq_len = min(
                min_seq_len + (global_step // 100) * 32,  # Increase by 32 every 100 steps
                max_seq_len  # Cap at max_seq_len
            )

            # Adjust batch size to maintain constant compute, but not below min_batch_size
            current_batch_size = max(min_batch_size, tokens_per_batch // current_seq_len)

            # Print when sequence length or batch size changes
            if current_seq_len != prev_len:
                print(f"\nStep {global_step}:")
                print(f"Sequence length set to {current_seq_len}")
                print(f"Batch size adjusted to {current_batch_size}")
                print(f"Tokens per batch: {current_seq_len * current_batch_size:,}")

                # Create new dataloader with updated batch size
                dataloader = DataLoader(
                    dataset,
                    batch_size=current_batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                )
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

                prev_len = current_seq_len
                prev_bs = current_batch_size

            batch = {k: v[:, :current_seq_len].to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Create attention mask for the model
            mask = None
            if "attention_mask" in batch:
                mask = batch["attention_mask"].unsqueeze(1).unsqueeze(2)
                mask = mask @ mask.transpose(-2, -1)

            # Forward pass with mask
            logits = model(batch["input_ids"], mask=mask)

            # Calculate loss (ignore padding tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = loss.item()

            # Log metrics
            wandb.log(
                {
                    "loss": loss_value,
                    "perplexity": math.exp(loss_value),
                    "epoch": epoch,
                    "global_step": global_step,
                }
            )

            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            # Generate sample periodically
            if global_step > 0 and global_step % sample_every == 0:
                sample = generate_sample(model, tokenizer, device, test_prompt)
                print(f"\nStep {global_step}, Loss {loss_value:.4f}")
                print(f"Sample: {sample}\n")
                wandb.log({"sample_text": sample, "step": global_step})

            # Save checkpoint periodically
            if global_step > 0 and global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, loss_value)

            global_step += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", action="store_true", help="Load checkpoint if it exists"
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        help="LoRA rank (default: hidden_size/8)",
        default=None,
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load models
    model, tokenizer, _ = load_model_and_tokenizer(lora_rank=args.rank)
    if model is None or tokenizer is None:
        print("Failed to load model properly")
        return

    # Train the model
    train_model(model, tokenizer, load_checkpoint_if_exists=args.load)


if __name__ == "__main__":
    main()
