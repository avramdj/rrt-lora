# Relaxed Recursive Transformers
Implementation of [RRT-LoRA](https://arxiv.org/abs/2410.20672) by Google DeepMind on TinyLlama.
<img width="1040" height="326" alt="Screenshot 2025-08-15 at 3 28 04 AM" src="https://github.com/user-attachments/assets/aa099b32-4e25-4a9c-bdf0-4b0c8f4bd60f" />

## basics
regular transformer:

$h_t^l = f(h_t^{l-1}; \Phi^l)$

recursive version with L layers and B blocks:

$h_t^l = f(h_t^{l-1}; \Phi^\prime_{((l-1) \bmod L/B + 1)})$

## relaxed version with learnable $W^\prime$ shared representation and LoRA
the rrt:

$h_t^l = f(h_t^{l-1}; \Phi^\prime_{((l-1) \bmod L/B + 1)}, \Delta\Phi^{\prime l})$

for each weight matrix at each layer:

$h = W^\prime x + BAx$ where:
- $W^\prime$ is learned shared weights
- $BA$ is position-specific LoRA (initialized via SVD)

## init and training process
1. compute residuals between original and tied for each position:

   $R^l = W^l - W^\prime_{((l-1) \bmod L/B + 1)}$
2. get initial LoRA weights via truncated SVD:

   $U_r^l, \Sigma_r^l, V_r^l = \text{TruncatedSVD}(R^l; r)$
   - $B^l = U_r^l \Sigma_r^l$ 
   - $A^l = (V_r^l)^T$

3. during training:
   - forward: $h = W^\prime x + B^lA^lx$ 
   - backward: update BOTH $W^\prime$ AND $B^l,A^l$ matrices
   - $W^\prime$ learns optimal shared representation
   - $B^l,A^l$ learn position-specific adjustments

so the final learned mapping approximates:

$W^l \approx W^\prime_{((l-1) \bmod L/B + 1)} + B^lA^l$

## References

```bibtex
@inproceedings{Bae2025Relaxed,
    title={Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA},
    author={Sangmin Bae and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and Seungyeon Kim and Tal Schuster},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

## How to Cite
If you use this implementation in your research, please cite it as follows:

```bibtex
@software{avram_rrt_lora_2024,
  author = {Your Name or Username},
  title = {rrt-lora: An Implementation of Relaxed Recursive Transformers},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/avramdj/rrt-lora}
}
```
