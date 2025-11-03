# LNN Mathematical Formulations

This document summarizes the core mathematical formulations implemented in the Fortran LNN library.

## Notation

- Scalars: lower-case (e.g., `a`), vectors: bold lower-case (e.g., `u`), matrices: bold upper-case (e.g., `U`), tensors: calligraphic (e.g., `T`).
- CP rank `R`, tensor dimensions `n1, n2, n3`.
- Khatri–Rao product: `A ⊙ B` (column-wise Kronecker).
- Hadamard product: `A ◦ B`.
- Frobenius norm: `||·||_F`.

## CP Decomposition (CANDECOMP/PARAFAC)

Given a 3-way tensor `T ∈ R^{n1×n2×n3}`, the CP decomposition seeks factor matrices `U ∈ R^{n1×R}`, `V ∈ R^{n2×R}`, `W ∈ R^{n3×R}` (and optionally diagonal weights `λ ∈ R^R`) such that

```
T ≈ ∑_{r=1}^R λ_r · u_r ⊗ v_r ⊗ w_r.
```

### Alternating Least Squares (ALS)

ALS updates one factor at a time by fixing the other two. Using normal equations, the update for `U` (mode-1) solves

```
( (V^T V) ◦ (W^T W) + λ·I ) · U^T = MTTKRP_mode1(T; V, W),
```

where `MTTKRP_mode1` is the Matricized-Tensor Times Khatri–Rao Product:

```
MTTKRP_mode1(i, r) = ∑_{j,k} T(i,j,k) · V(j,r) · W(k,r).
```

Analogous updates hold for `V` and `W` using modes 2 and 3. We solve the SPD systems via Cholesky (DPOTRF/DPOTRS) with a small Tikhonov regularization added to the Gram diagonal.

### Scaling and Normalization

Factor columns are normalized after each update; the scaling is accumulated into `λ` such that the product `λ_r · u_r ⊗ v_r ⊗ w_r` is invariant.

### Fit and Residual

Fit with respect to Frobenius norm is

```
fit = 1 − ||T − \hat{T}||_F^2 / ||T||_F^2,
```

where `\hat{T} = ∑_r λ_r · u_r ⊗ v_r ⊗ w_r`.

## SPD Manifold Operations

For SPD matrices `A,B ∈ S_{++}^n`:

- Log map at `A` of `V`:
  1. Eigendecompose `A = QΛQ^T` with `Λ = diag(λ_i)` and floor eigenvalues: `λ_i ← max(λ_i, ε)`.
  2. Compute `A^{±1/2} = Q·diag(λ_i^{±1/2})·Q^T`.
  3. `Y = A^{-1/2} V A^{-1/2}`; eigendecompose `Y = Q_Y Λ_Y Q_Y^T`.
  4. `log_A(V) = A^{1/2} · (Q_Y log(Λ_Y) Q_Y^T) · A^{1/2}`.

- Exp map similar, using `exp(Λ_Y)` in step 4.

- Affine-invariant distance:

```
dist(A,B) = || log(A^{-1/2} B A^{-1/2}) ||_F = (∑_i log^2(μ_i))^{1/2}
```

where `μ_i` are eigenvalues of `A^{-1/2} B A^{-1/2}`.

We compute eigendecompositions via DSYEVD (divide & conquer), with optional DSYEVR (MRRR) path for clustered spectra.

## Hyperbolic (Poincaré Ball) Operations

In unit ball model with curvature −1, vectors `x, y ∈ B^n`:

- Möbius addition (implemented in `lnn_mobius_add`) and exponential map are provided for basic hyperbolic updates; for stability near the boundary, an `EPS` floor is used.

## Similarity and Top‑K Search

- Cosine similarity: `cos(a,b) = a·b / (||a||·||b||)` with clamping to [−1,1].
- Batched norms and fused cosine reduce redundant work.
- Top‑K via min-heap: maintain a size‑K min-heap of scores; replace root when a new score exceeds it, then sift‑down; finish with heap-sort to obtain descending order.

## Numerical Stability

- Kahan summation for layernorm and tensor contractions.
- Log-sum-exp softmax.
- Eigenvalue flooring in SPD ops.
- Regularization in ALS Gram systems.

