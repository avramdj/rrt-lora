import argparse
import gc
import math
import os
from time import time
from typing import Any, Callable, Optional, TypeVar

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from model import RecursiveTinyLlama

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ORIGINAL_PARAMS: Optional[int] = None

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def compile(
    model: Callable[[InputT], OutputT], *args, **kwargs
) -> Callable[[InputT], OutputT]:
    if get_device() == torch.device("cuda"):
        return torch.compile(model, *args, **kwargs)
    return model


def calculate_backbone_and_lora_inits(
    hf_model: nn.Module,
    num_layers: int,
    num_blocks: int,
    lora_rank: int,
) -> tuple[
    list[dict[str, dict[str, torch.Tensor]]], dict[str, list[dict[str, torch.Tensor]]]
]:
    original_weights = {
        "self_attn": {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": []},
        "mlp": {"gate_proj": [], "up_proj": [], "down_proj": []},
    }
    for name, param in hf_model.named_parameters():
        if name.startswith("model.layers"):
            parts = name.split(".")
            try:
                layer_idx, block_type, proj_type = int(parts[2]), parts[3], parts[4]
                while len(original_weights[block_type][proj_type]) <= layer_idx:
                    original_weights[block_type][proj_type].append(None)
                original_weights[block_type][proj_type][layer_idx] = param.data.clone()
            except (KeyError, IndexError):
                continue

    backbone_weights = [
        {
            "self_attn": {
                "q_proj": None,
                "k_proj": None,
                "v_proj": None,
                "o_proj": None,
            },
            "mlp": {"gate_proj": None, "up_proj": None, "down_proj": None},
        }
        for _ in range(num_blocks)
    ]

    for k in range(num_blocks):
        for block_type, projs in original_weights.items():
            for proj_type, W_l_list in projs.items():
                grouped_layers = [
                    W_l_list[layer_idx]
                    for layer_idx in range(k, num_layers, num_blocks)
                ]
                avg_weight = torch.stack(grouped_layers, dim=0).mean(dim=0)
                backbone_weights[k][block_type][proj_type] = avg_weight

    lora_initial_weights = [
        {
            "self_attn": {
                "q_proj": None,
                "k_proj": None,
                "v_proj": None,
                "o_proj": None,
            },
            "mlp": {"gate_proj": None, "up_proj": None, "down_proj": None},
        }
        for _ in range(num_layers)
    ]

    if lora_rank > 0:
        for layer_idx in range(num_layers):
            k = layer_idx % num_blocks
            for block_type, projs in original_weights.items():
                for proj_type, W_l_list in projs.items():
                    W_l = W_l_list[layer_idx]
                    W_k_prime = backbone_weights[k][block_type][proj_type]
                    residual = W_l - W_k_prime

                    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                    U_r, S_r, Vh_r = U[:, :lora_rank], S[:lora_rank], Vh[:lora_rank, :]

                    B = U_r @ torch.diag(S_r)
                    A = Vh_r

                    lora_initial_weights[layer_idx][block_type][proj_type] = {
                        "A": A,
                        "B": B,
                    }

    return backbone_weights, lora_initial_weights


def load_model_and_tokenizer(
    lora_rank: int,
    num_blocks: int,
    store_original: bool = False,
) -> tuple[
    Optional[RecursiveTinyLlama], Optional[Any], Optional[dict[str, torch.Tensor]]
]:
    global ORIGINAL_PARAMS
    device = get_device()

    hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name).to(device)
    # print(hf_model.model)
    ORIGINAL_PARAMS = sum(p.numel() for p in hf_model.parameters())
    print(f"Original parameters: {ORIGINAL_PARAMS:,}")

    config = hf_model.config

    full_rank = config.hidden_size

    print("\nWeight matrix dimensions:")
    print(f"Full rank: {full_rank}")
    print(f"LoRA rank: {lora_rank}")

    print("Calculating backbone and LoRA initial weights...")
    backbone_weights, lora_initial_weights = calculate_backbone_and_lora_inits(
        hf_model, config.num_hidden_layers, num_blocks, lora_rank
    )
    print("Backbone and LoRA initial weights calculated")

    model = RecursiveTinyLlama(
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        num_key_value_heads=config.num_key_value_heads,
        num_blocks=num_blocks,
        lora_rank=lora_rank,
        init_lora_weights=lora_initial_weights,
        backbone_weights=backbone_weights,
    ).to(device)

    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(hf_model.model.embed_tokens.weight)
        model.get_output_embeddings().weight.copy_(hf_model.lm_head.weight)

    print("Compiling model...")
    begin = time()
    model = compile(model, fullgraph=False, mode="max-autotune-no-cudagraphs")
    end = time()
    print(f"Model compiled in {end - begin:.2f} seconds")

    original_weights: Optional[dict[str, torch.Tensor]] = {} if store_original else None

    del config
    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
) -> tuple[int, float]:
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
        use_cuda_amp = device.type == "cuda"
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp
        ):
            generated = model.generate(
                tokens, max_new_tokens=64, temperature=0.7, top_k=50
            )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
    model.train()
    return response


def generate_samples(
    model: RecursiveTinyLlama, tokenizer: Any, device: torch.device
) -> list[tuple[str, str]]:
    """Generate multiple sample prompts and completions."""
    sample_prompts: list[str] = [
        "The capital of France is Paris, and the capital of Germany is",
        "Write a Python function that reverses a string:",
        "Translate to French: 'How are you today?'",
        "Summarize in one sentence: Large Language Models are changing software development by",
        "Continue this story: Once upon a time, in a quiet village,",
    ]
    results: list[tuple[str, str]] = []
    for prompt in sample_prompts:
        completion = generate_sample(model, tokenizer, device, prompt)
        results.append((prompt, completion))
    return results


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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
    learning_rate: float = 3e-4,
    load_checkpoint_if_exists: bool = False,
    save_every: int = 500,
    sample_every: int = 100,
    grad_accumulation_steps: int = 8,
) -> None:
    """Train the model on wikitext data"""
    device = get_device()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_rank = model.blocks[0].attention.wq.lora.rank

    print("\nParameter counts:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA rank: {lora_rank}")

    print(f"Original parameters: {ORIGINAL_PARAMS:,}")
    print(f"Reduction: {100 * (1 - total_params / ORIGINAL_PARAMS):.2f}%")

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
            "lora_rank": lora_rank,
            "compression_ratio": lora_rank
            / model.blocks[0].attention.wq.weight.shape[0],  # r/d ratio
            "grad_accumulation_steps": grad_accumulation_steps,
        },
    )

    dataset = TextDataset(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if "lora" in n],
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if "lm_head" in n],
                "lr": learning_rate / 4,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if ("weight" in n and "lora" not in n and "lm_head" not in n)
                ],
                "lr": learning_rate / 4,
            },
        ]
    )

    # Scheduler: 500-step warmup then linear decay to 0
    steps_per_epoch = max(1, len(dataloader) // grad_accumulation_steps)
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = 500

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if load_checkpoint_if_exists:
        global_step, best_loss = load_checkpoint(model, optimizer)

    optimizer.zero_grad()
    progress_bar = tqdm(total=total_steps, initial=global_step, desc="Training (steps)")
    finished = False
    for epoch in range(start_epoch, num_epochs):
        if finished:
            break
        model.train()
        accumulated_loss = 0.0
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            use_cuda_amp = device.type == "cuda"
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp
            ):
                logits = model(batch["input_ids"])

            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            loss = loss / grad_accumulation_steps
            accumulated_loss += loss.item()
            # Backward pass
            loss.backward()

            if (i + 1) % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                loss_value = accumulated_loss
                accumulated_loss = 0.0

                # Step LR scheduler AFTER optimizer step
                scheduler.step()

                # Increment global step and update progress bar
                global_step += 1
                progress_bar.update(1)

                # Log metrics
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "loss": loss_value,
                        "perplexity": math.exp(loss_value),
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": current_lr,
                    }
                )
                progress_bar.set_postfix(
                    {"loss": f"{loss_value:.4f}", "lr": f"{current_lr:.2e}"}
                )

                # Generate multiple samples periodically
                if global_step > 0 and global_step % sample_every == 0:
                    samples = generate_samples(model, tokenizer, device)
                    print(f"\nStep {global_step}, Loss {loss_value:.4f}")
                    for idx, (prompt_text, completion) in enumerate(samples):
                        print(f"Prompt {idx + 1}: {prompt_text}")
                        print(f"Completion {idx + 1}: {completion}\n")
                    samples_table = wandb.Table(columns=["idx", "prompt", "output"])
                    for idx, (p, o) in enumerate(samples):
                        samples_table.add_data(idx, p, o)
                    wandb.log({"samples_table": samples_table, "step": global_step})

                # Save checkpoint periodically
                if global_step > 0 and global_step % save_every == 0:
                    save_checkpoint(model, optimizer, global_step, loss_value)

                # Stop if we have reached total steps
                if global_step >= total_steps:
                    finished = True
                    break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", action="store_true", help="Load checkpoint if it exists"
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        help="LoRA rank",
        default=256,
    )
    parser.add_argument(
        "-g",
        "--grad-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "-b",
        "--num-blocks",
        type=int,
        help="Number of blocks",
        default=2,
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load models
    model, tokenizer, _ = load_model_and_tokenizer(
        lora_rank=args.rank, num_blocks=args.num_blocks
    )
    if model is None or tokenizer is None:
        print("Failed to load model properly")
        return

    # Train the model
    train_model(
        model,
        tokenizer,
        load_checkpoint_if_exists=args.load,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )


if __name__ == "__main__":
    main()
