import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Dict, Any
from model import RecursiveTinyLlama
import wandb
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer() -> (
    Tuple[
        Optional[RecursiveTinyLlama], Optional[Any], Optional[Dict[str, torch.Tensor]]
    ]
):
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
    lora_rank = full_rank // 8  # The reduced rank we'll use
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
        lora_rank=lora_rank,  # Pass the rank to the model
    ).to(device)

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


def train_weight_matching(
    model: RecursiveTinyLlama,
    original_weights: Dict[str, torch.Tensor],
    tokenizer: Any,
    num_iterations: int = 10000,
    learning_rate: float = 1e-4,
    load_checkpoint_if_exists: bool = False,
    checkpoint_every: int = 1000,
    sample_every: int = 500,
) -> None:
    """Train shared weights W' and LoRA weights BA to match original layer weights"""
    device = get_device()
    model = model.to(device)

    # Print LoRA parameter counts
    total_params = 0
    lora_params = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_params += param.numel()
        total_params += param.numel()

    print(f"\nParameter counts:")
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")

    wandb.init(
        project="relaxed-recursive-tinyllama",
        config={
            "num_iterations": num_iterations,
            "learning_rate": learning_rate,
            "num_blocks": model.num_blocks,
            "num_layers": model.n_layers,
            "lora_rank": model.blocks[0].attention.wq.lora.rank,  # Log LoRA rank
            "total_params": total_params,
            "lora_params": lora_params,
        },
    )

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

    start_step = 0
    best_loss = float("inf")

    if load_checkpoint_if_exists:
        start_step, best_loss = load_checkpoint(model, optimizer)

    progress_bar = tqdm(range(start_step, num_iterations), desc="Training")

    test_prompt = "The capital of France is Paris, and the capital of Germany is"

    for step in progress_bar:
        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        for block_idx, block in enumerate(model.blocks):
            layer_indices = [
                i for i in range(model.n_layers) if i % model.num_blocks == block_idx
            ]

            for layer_idx in layer_indices:
                for name, module in [
                    ("q", block.attention.wq),
                    ("k", block.attention.wk),
                    ("v", block.attention.wv),
                    ("o", block.attention.wo),
                ]:
                    orig_key = f"layer_{layer_idx}_{name}"
                    if orig_key not in original_weights:
                        continue

                    orig_weight = original_weights[orig_key].to(device)
                    current_weight = module.weight + (module.lora.B @ module.lora.A)

                    loss = F.mse_loss(current_weight, orig_weight)
                    total_loss += loss

        total_loss.backward()
        optimizer.step()

        # Log metrics
        loss_value = total_loss.item()
        wandb.log({"total_loss": loss_value, "step": step})

        progress_bar.set_postfix({"loss": loss_value})

        # Generate sample every sample_every steps
        if step > 0 and step % sample_every == 0:
            sample = generate_sample(model, tokenizer, device, test_prompt)
            print(f"\nStep {step}, Loss {loss_value:.4f}")
            print(f"Sample: {sample}\n")
            wandb.log({"sample_text": sample, "step": step})

        # Save checkpoint every checkpoint_every steps if loss improved
        if step > 0 and step % checkpoint_every == 0 and loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(model, optimizer, step, loss_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", action="store_true", help="Load checkpoint if it exists"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer, original_weights = load_model_and_tokenizer()
    if model is None or tokenizer is None or original_weights is None:
        print("Failed to load model properly")
        return

    train_weight_matching(
        model, original_weights, tokenizer, load_checkpoint_if_exists=args.load
    )

    # Final test generation
    sample = generate_sample(
        model,
        tokenizer,
        device,
        "The capital of France is Paris, and the capital of Germany is",
    )
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
