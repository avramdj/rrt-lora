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


def load_model_and_tokenizer(
    lora_rank: Optional[int] = None,
) -> Tuple[
    Optional[RecursiveTinyLlama], Optional[Any], Optional[Dict[str, torch.Tensor]]
]:
    device = get_device()

    # First load the HF model and tokenizer
    hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name).to(device)
    print(hf_model.model)

    # Create our custom model with same config
    config = hf_model.config

    # Calculate and print the full rank size
    full_rank = config.hidden_size  # This is the dimension of the weight matrices
    if lora_rank is None:
        lora_rank = full_rank // 8  # Default to hidden_size/8 if not specified

    print("\nWeight matrix dimensions:")
    print(f"Full rank: {full_rank} x {full_rank}")
    print(f"LoRA rank: {lora_rank} (reduction: {lora_rank/full_rank*100:.2f}%)")

    model = RecursiveTinyLlama(
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        num_key_value_heads=config.num_key_value_heads,
        num_blocks=4,
        lora_rank=lora_rank,
    ).to(device)

    print("Compiling model...")
    begin = time()
    model = torch.compile(model, fullgraph=True, mode="max-autotune")
    end = time()
    print(f"Model compiled in {end - begin:.2f} seconds")

    # Store original weights with better keys
    original_weights: Dict[str, torch.Tensor] = {}

    def copy_weights(
        m1: torch.nn.Module, m2: torch.nn.Module, layer_idx: int, weight_type: str
    ) -> bool:
        with torch.no_grad():
            if m1.weight.shape != m2.weight.shape:
                print(f"Shape mismatch: {m1.weight.shape} vs {m2.weight.shape}")
                return False
            key = f"layer_{layer_idx}_{weight_type}"
            original_weights[key] = m2.weight.clone()
            m1.weight.copy_(m2.weight)
            return True

    # Copy transformer layers with proper indexing
    for block_idx, block in enumerate(model.blocks):
        # Copy weights from corresponding HF layer
        l2 = hf_model.model.layers[block_idx]
        if not all(
            [
                copy_weights(block.attention.wq, l2.self_attn.q_proj, block_idx, "q"),
                copy_weights(block.attention.wk, l2.self_attn.k_proj, block_idx, "k"),
                copy_weights(block.attention.wv, l2.self_attn.v_proj, block_idx, "v"),
                copy_weights(block.attention.wo, l2.self_attn.o_proj, block_idx, "o"),
            ]
        ):
            print(f"Error in attention weight copying for block {block_idx}")
            return None, None, None

    print("Model loaded successfully!")
    return model, hf_tokenizer, original_weights


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

        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - max_length, max_length // 2):
            self.sequences.append(all_tokens[i : i + max_length])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def train_model(
    model: RecursiveTinyLlama,
    tokenizer: Any,
    num_epochs: int = 25,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    load_checkpoint_if_exists: bool = False,
    save_every: int = 500,
    sample_every: int = 500,
) -> None:
    """Train the model on wikitext data"""
    device = get_device()
    model = model.to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_rank = model.blocks[0].attention.wq.lora.rank  # Get LoRA rank from model

    print(f"\nParameter counts:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA rank: {lora_rank}")

    # Initialize wandb
    wandb.init(
        project="relaxed-recursive-tinyllama",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
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
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
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
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            logits = model(batch["input_ids"])

            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
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
