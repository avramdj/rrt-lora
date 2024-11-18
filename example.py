import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import TinyLlama

def load_model_and_tokenizer():
    # First load the HF model and tokenizer
    hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    print(hf_model.model)
    # Create our custom model with same config
    config = hf_model.config
    model = TinyLlama(
        vocab_size=config.vocab_size,
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        num_key_value_heads=config.num_key_value_heads
    )
    
    # Copy weights from HF model to our model
    def copy_weights(m1, m2):
        with torch.no_grad():
            if m1.weight.shape != m2.weight.shape:
                print(f"Shape mismatch: {m1.weight.shape} vs {m2.weight.shape}")
                return False
            m1.weight.copy_(m2.weight)
            return True
    
    # Copy embeddings
    copy_weights(model.token_embedding, hf_model.model.embed_tokens)
    
    # Copy transformer layers
    for l1, l2 in zip(model.layers, hf_model.model.layers):
        # Attention weights
        if not all([
            copy_weights(l1.attention.wq, l2.self_attn.q_proj),
            copy_weights(l1.attention.wk, l2.self_attn.k_proj),
            copy_weights(l1.attention.wv, l2.self_attn.v_proj),
            copy_weights(l1.attention.wo, l2.self_attn.o_proj)
        ]):
            print("Error in attention weight copying")
            return None, None
        
        # FFN weights
        if not all([
            copy_weights(l1.feed_forward.gate_proj, l2.mlp.gate_proj),
            copy_weights(l1.feed_forward.down_proj, l2.mlp.down_proj),
            copy_weights(l1.feed_forward.up_proj, l2.mlp.up_proj)
        ]):
            print("Error in FFN weight copying")
            return None, None
        
        # Norms
        if not all([
            copy_weights(l1.attention_norm, l2.input_layernorm),
            copy_weights(l1.ffn_norm, l2.post_attention_layernorm)
        ]):
            print("Error in norm weight copying")
            return None, None
    
    if not all([
        copy_weights(model.norm, hf_model.model.norm),
        copy_weights(model.lm_head, hf_model.lm_head)
    ]):
        print("Error in final layer copying")
        return None, None
    
    print("Model loaded successfully!")
    return model, hf_tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("Failed to load model properly")
        return
        
    model.eval()
    
    prompt = "The capital of France is Paris, and the capital of Germany is"
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    
    generated = model.generate(
        tokens,
        max_new_tokens=20,
        temperature=0.7,
        top_k=50
    )
    
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()
