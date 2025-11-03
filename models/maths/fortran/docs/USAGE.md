# LNN Usage Guide

This guide shows common workflows using the Fortran LNN library.

## Building

- Default (no BLAS/LAPACK):
  - `make lib FC=gfortran`
- With OpenBLAS + LAPACK (Linux):
  - `make lib FC=gfortran LNN_USE_BLAS=1 LNN_USE_LAPACK=1`
- Enable DSYEVR (MRRR) eigensolver path:
  - add `LNN_USE_DSYEVR=1`

See `scripts/build_linux.sh` for a ready-to-use build script on x86_64.

## CP Decomposition (ALS)

Example: decompose a synthetic tensor to rank R, returning factors and weights λ.

```
program example_cp
  use iso_c_binding
  use advanced_als
  implicit none
  integer(c_int), parameter :: n1=30, n2=20, n3=10, R=5, max_iters=50
  real(c_double), allocatable :: T(:), U(:), V(:), W(:), lambda(:), fit(:)
  integer(c_int) :: iters
  integer :: i

  allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R), fit(max_iters))
  call random_number(T); call random_number(U); call random_number(V); call random_number(W)

  call lnn_cp_als_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda, 1.0d-6, max_iters, fit, iters)
  call lnn_cp_print_fit_trace(fit, iters)
end program example_cp
```

See `examples/cp_decomposition.f90`.

## SPD Operations

Compute affine-invariant distance and exp/log maps:

```
program example_spd
  use iso_c_binding
  use tensor_kernels
  implicit none
  integer(c_int), parameter :: n=3
  real(c_double) :: A(9), B(9), V(9), out(9), dist
  ! Fill SPD matrices A, B and symmetric V ...
  call lnn_spd_distance(n, A, B, dist)
  call lnn_spd_exp_log(n, A, V, out, 0_c_int) ! exp_A(V)
  call lnn_spd_exp_log(n, A, out, V, 1_c_int) ! log_A(exp_A(V)) ≈ V
end program example_spd
```

See `examples/spd_ops.f90`.

## Top‑K Similarity Search

Compute batched norms and run optimized cosine Top‑K with precomputed norms:

```
program example_topk
  use iso_c_binding
  use lnn_kernels
  implicit none
  integer(c_int), parameter :: n=128, m=10000, k=10
  real(c_double), allocatable :: A(:), norms(:), q(:), scores(:)
  integer(c_int), allocatable :: idx(:)
  real(c_double) :: qn
  integer :: i

  allocate(A(n*m), norms(m), q(n), scores(k), idx(k))
  call random_number(A); call random_number(q)
  call lnn_compute_norms(n, m, A, norms)
  call lnn_vector_norm(n, q, qn)
  call lnn_cosine_topk_optimized(n, m, A, norms, q, qn, k, idx, scores)
end program example_topk
```

See `examples/topk_search.f90`.

## Profiling

Use benchmarks under `bench/` or enable instrumentation by building with `-DLNN_PROFILE` and using the profiler hooks (see `lnn_profiler.f90`).

