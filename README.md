# Relaxed Recursive Transformers
Implementation of [RRT-LoRA](https://arxiv.org/abs/2410.20672) by Google DeepMind on TinyLlama.

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

