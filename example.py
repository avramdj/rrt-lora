import os
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set tokenizers parallelism before importing/using tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TinyLlama:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None
    ):
        """
        Initialize TinyLlama model and tokenizer.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to place model on ('cuda', 'cpu', etc). If None, auto-detect.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a response given a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        # Format prompt using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return the generated text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main() -> None:
    """Main function to demonstrate TinyLlama chat capabilities."""
    # Initialize model
    model = TinyLlama()
    
    # Example conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant who provides clear and concise answers."
        },
        {
            "role": "user",
            "content": "What is the capital of France and what is it famous for?"
        }
    ]
    
    # Generate and print response
    print("Generating response...")
    response = model.generate(messages)
    print("\nFull conversation:")
    print(response)

if __name__ == "__main__":
    main() 