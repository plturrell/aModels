module tensor_kernels
  use iso_c_binding
  use omp_lib
  use lnn_config, only: DEFAULT_EPS, SPD_ABSTOL, SPD_EIG_PATH, SPD_EIG_DSYEVR, lnn_read_env
  implicit none
  
  ! Numerical stability constants
  real(c_double), parameter :: EPS = DEFAULT_EPS
  real(c_double), parameter :: LOG_EPS = -27.631021115928547d0  ! log(1e-12)

  ! Expose CP-ALS step via C binding; implemented in advanced_als
  interface
    subroutine lnn_cp_als_step(n1, n2, n3, R, T, U, V, W, lambda, which, out) &
        bind(C, name='lnn_cp_als_step')
      use iso_c_binding
      integer(c_int), value :: n1, n2, n3, R, which
      real(c_double), intent(in) :: T(*), U(*), V(*), W(*)
      real(c_double), value :: lambda
      real(c_double), intent(out) :: out(*)
    end subroutine lnn_cp_als_step
  end interface
  
contains

  ! ============================================================================
  ! BATCHED GEMM + SOFTMAX
  ! ============================================================================
  
  ! Batched matrix multiplication with optional softmax fusion
  subroutine lnn_gemm_batched(m, n, k, batch_size, A, B, C, apply_softmax) &
      bind(C, name='lnn_gemm_batched')
    integer(c_int), value :: m, n, k, batch_size
    real(c_double), intent(in) :: A(*)  ! batch_size x (m x k) column-major
    real(c_double), intent(in) :: B(*)  ! batch_size x (k x n) column-major
    real(c_double), intent(out) :: C(*) ! batch_size x (m x n) column-major
    integer(c_int), value :: apply_softmax
    
    integer :: bb, offset_a, offset_b, offset_c
    real(c_double) :: alpha, beta
    
    alpha = 1.0d0
    beta = 0.0d0
    
    !$OMP PARALLEL DO PRIVATE(bb, offset_a, offset_b, offset_c)
    do bb = 0, batch_size - 1
      offset_a = bb * m * k
      offset_b = bb * k * n
      offset_c = bb * m * n
      
#if defined(USE_BLAS) || defined(LNN_USE_BLAS)
      call dgemm('N', 'N', n, m, k, alpha, &
                 B(offset_b + 1), n, &
                 A(offset_a + 1), k, &
                 beta, C(offset_c + 1), n)
#else
      call naive_matmul(m, n, k, A(offset_a + 1), B(offset_b + 1), C(offset_c + 1))
#endif
      
      if (apply_softmax == 1) then
        call softmax_inplace(m, n, C(offset_c + 1))
      end if
    end do
    !$OMP END PARALLEL DO
  end subroutine lnn_gemm_batched
  
  ! Stable softmax using log-sum-exp trick
  subroutine lnn_softmax_batched(m, n, X, out) bind(C, name='lnn_softmax_batched')
    integer(c_int), value :: m, n
    real(c_double), intent(in) :: X(*)
    real(c_double), intent(out) :: out(*)
    
    integer :: i, j, idx
    real(c_double) :: max_val, sum_exp
    
    !$OMP PARALLEL DO PRIVATE(i, j, idx, max_val, sum_exp)
    do i = 0, m - 1
      ! Find max for numerical stability
      max_val = -huge(1.0d0)
      !$OMP SIMD
      do j = 0, n - 1
        idx = i * n + j + 1
        if (X(idx) > max_val) max_val = X(idx)
      end do
      
      ! Compute exp(x - max) and sum
      sum_exp = 0.0d0
      !$OMP SIMD
      do j = 0, n - 1
        idx = i * n + j + 1
        out(idx) = exp(X(idx) - max_val)
        sum_exp = sum_exp + out(idx)
      end do
      
      ! Normalize
      !$OMP SIMD
      do j = 0, n - 1
        idx = i * n + j + 1
        out(idx) = out(idx) / (sum_exp + EPS)
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine lnn_softmax_batched
  
  subroutine softmax_inplace(m, n, X)
    integer, intent(in) :: m, n
    real(c_double), intent(inout) :: X(n, m)
    integer :: i, j
    real(c_double) :: max_val, sum_exp
    
    do i = 1, m
      max_val = maxval(X(:, i))
      sum_exp = 0.0d0
      do j = 1, n
        X(j, i) = exp(X(j, i) - max_val)
        sum_exp = sum_exp + X(j, i)
      end do
      X(:, i) = X(:, i) / (sum_exp + EPS)
    end do
  end subroutine softmax_inplace
  
  ! ============================================================================
  ! LAYER NORMALIZATION
  ! ============================================================================
  
  ! LayerNorm with Kahan summation for stability
  subroutine lnn_layernorm(n, X, gamma, beta, eps, out) bind(C, name='lnn_layernorm')
    integer(c_int), value :: n
    real(c_double), intent(in) :: X(*), gamma(*), beta(*)
    real(c_double), value :: eps
    real(c_double), intent(out) :: out(*)
    
    real(c_double) :: mean, variance, std_dev
    real(c_double) :: sum_val, sum_sq, c, y, t
    integer :: i
    
    ! Kahan summation for mean
    sum_val = 0.0d0
    c = 0.0d0
    do i = 1, n
      y = X(i) - c
      t = sum_val + y
      c = (t - sum_val) - y
      sum_val = t
    end do
    mean = sum_val / real(n, c_double)
    
    ! Kahan summation for variance
    sum_sq = 0.0d0
    c = 0.0d0
    do i = 1, n
      y = (X(i) - mean)**2 - c
      t = sum_sq + y
      c = (t - sum_sq) - y
      sum_sq = t
    end do
    variance = sum_sq / real(n, c_double)
    std_dev = sqrt(variance + eps)
    
    ! Normalize and scale
    !$OMP SIMD
    do i = 1, n
      out(i) = gamma(i) * (X(i) - mean) / std_dev + beta(i)
    end do
  end subroutine lnn_layernorm
  
  ! RMSNorm (Root Mean Square Normalization)
  subroutine lnn_rmsnorm(n, X, gamma, eps, out) bind(C, name='lnn_rmsnorm')
    integer(c_int), value :: n
    real(c_double), intent(in) :: X(*), gamma(*)
    real(c_double), value :: eps
    real(c_double), intent(out) :: out(*)
    
    real(c_double) :: rms, sum_sq, c, y, t
    integer :: i
    
    ! Kahan summation for sum of squares
    sum_sq = 0.0d0
    c = 0.0d0
    do i = 1, n
      y = X(i)**2 - c
      t = sum_sq + y
      c = (t - sum_sq) - y
      sum_sq = t
    end do
    
    rms = sqrt(sum_sq / real(n, c_double) + eps)
    
    !$OMP SIMD
    do i = 1, n
      out(i) = gamma(i) * X(i) / rms
    end do
  end subroutine lnn_rmsnorm
  
  ! ============================================================================
  ! TENSOR CONTRACTIONS
  ! ============================================================================
  
  ! 3-way tensor contraction: result = sum_ijk A(i,j,k) * B(i) * C(j) * D(k)
  subroutine lnn_tensor_contract_3(nA1, nA2, nA3, A, nB, B, nC, C, nD, D, result) &
      bind(C, name='lnn_tensor_contract_3')
    integer(c_int), value :: nA1, nA2, nA3, nB, nC, nD
    real(c_double), intent(in) :: A(*), B(*), C(*), D(*)
    real(c_double), intent(out) :: result
    
    integer :: i, j, k, idx
    real(c_double) :: sum_val, c_kahan, y, t
    
    sum_val = 0.0d0
    c_kahan = 0.0d0
    
    do k = 1, nA3
      do j = 1, nA2
        do i = 1, nA1
          idx = i + (j-1)*nA1 + (k-1)*nA1*nA2
          y = A(idx) * B(i) * C(j) * D(k) - c_kahan
          t = sum_val + y
          c_kahan = (t - sum_val) - y
          sum_val = t
        end do
      end do
    end do
    
    result = sum_val
  end subroutine lnn_tensor_contract_3
  
  ! ============================================================================
  ! CP DECOMPOSITION (CANDECOMP/PARAFAC) - ALS
  ! ============================================================================
  
  ! lnn_cp_als_step: One ALS step for CP decomposition
  !
  ! Inputs:
  !   n1, n2, n3 - tensor dimensions
  !   R - rank of decomposition
  !   T - input tensor (n1 x n2 x n3)
  !   U, V, W - factor matrices
  !   lambda - regularization parameter
  !   which - which factor to update (0=U, 1=V, 2=W)
  !
  ! Output:
  !   out - updated factor matrix
  !
  ! Note: The lnn_cp_als_step implementation is provided in advanced_als
  ! via the same bind(C) symbol; this module exposes the interface above.
  
  ! ============================================================================
  ! SPD MANIFOLD OPERATIONS
  ! ============================================================================
  
  ! SPD matrix exponential/logarithm with full LAPACK
  subroutine lnn_spd_exp_log(n, A, V, out, mode) bind(C, name='lnn_spd_exp_log')
    integer(c_int), value :: n, mode
    real(c_double), intent(in) :: A(*), V(*)
    real(c_double), intent(out) :: out(*)
    
    ! mode: 0 = exp_A(V), 1 = log_A(V)
    real(c_double) :: work_a(n, n), work_v(n, n), temp(n, n)
    real(c_double) :: eigenvalues(n), eigenvectors(n, n)
    real(c_double) :: sqrt_a(n, n), inv_sqrt_a(n, n)
    real(c_double), allocatable :: work(:)
    integer, allocatable :: iwork(:)
    ! Workspace query helpers (used for DSYEVD/DSYEVR paths)
    real(c_double) :: work_q(1), vl, vu, abstol
    integer :: iwork_q(1), m
    integer, allocatable :: isuppz(:)
    integer :: isuppz_q(2)
    real(c_double) :: lambda_mod(n)
    integer :: i, j, k, info, lwork, liwork
    character :: jobz, uplo
    
    ! Symmetrize A
    call lnn_read_env()
    do j = 1, n
      do i = 1, n
        work_a(i, j) = 0.5d0 * (A((j-1)*n + i) + A((i-1)*n + j))
      end do
    end do
    
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
    ! Eigendecomposition: A = Q * Λ * Q^T
    jobz = 'V'
    uplo = 'U'
    eigenvectors = work_a
#ifdef LNN_USE_DSYEVR
    if (SPD_EIG_PATH == SPD_EIG_DSYEVR) then
      ! DSYEVR (MRRR) path
      vl = 0.0d0; vu = 0.0d0; abstol = SPD_ABSTOL
      call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                  eigenvectors, n, isuppz_q, work_q, -1, iwork_q, -1, info)
      lwork = int(work_q(1)); liwork = iwork_q(1)
      allocate(work(lwork)); allocate(iwork(liwork)); allocate(isuppz(2*n))
      call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                  eigenvectors, n, isuppz, work, lwork, iwork, liwork, info)
      deallocate(isuppz)
    else
      ! DSYEVD path
      call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work_q, -1, iwork_q, -1, info)
      lwork = int(work_q(1)); liwork = iwork_q(1)
      allocate(work(lwork)); allocate(iwork(liwork))
      call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
    end if
#else
    ! DSYEVD path (DSYEVR unavailable at build time)
    call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work_q, -1, iwork_q, -1, info)
    lwork = int(work_q(1)); liwork = iwork_q(1)
    allocate(work(lwork)); allocate(iwork(liwork))
    call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
#endif
    
    if (info /= 0) then
      write(*,*) 'LAPACK dsyev failed in lnn_spd_exp_log (A eig) with info =', info
      do i = 1, n*n
        out(i) = A(i)
      end do
      return
    end if
    
    ! Ensure positive definiteness (eigenvalue floor)
    do i = 1, n
      eigenvalues(i) = max(eigenvalues(i), EPS)
    end do
    
    if (mode == 0) then
      ! Exponential map: exp_A(V) = A^{1/2} * exp(A^{-1/2} V A^{-1/2}) * A^{1/2}
      
      ! Compute A^{1/2} and A^{-1/2}
      do i = 1, n
        lambda_mod(i) = sqrt(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, sqrt_a)
      
      do i = 1, n
        lambda_mod(i) = 1.0d0 / sqrt(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, inv_sqrt_a)
      
      ! Compute A^{-1/2} V A^{-1/2}
      call spd_triple_product(n, inv_sqrt_a, V, inv_sqrt_a, temp)
      
      ! Eigendecompose temp and apply exp
      eigenvectors = temp
#ifdef LNN_USE_DSYEVR
      if (SPD_EIG_PATH == SPD_EIG_DSYEVR) then
        allocate(isuppz(2*n))
        call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                    eigenvectors, n, isuppz, work, lwork, iwork, liwork, info)
        deallocate(isuppz)
      else
        call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
      end if
#else
      call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
#endif
      if (info /= 0) then
        write(*,*) 'LAPACK dsyev failed in lnn_spd_exp_log (temp eig, exp) info =', info
        do i = 1, n*n
          out(i) = A(i)
        end do
        return
      end if
      
      do i = 1, n
        lambda_mod(i) = exp(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, temp)
      
      ! Final: A^{1/2} * temp * A^{1/2}
      call spd_triple_product(n, sqrt_a, temp, sqrt_a, work_a)
      
    else
      ! Logarithm map: log_A(V) = A^{1/2} * log(A^{-1/2} V A^{-1/2}) * A^{1/2}
      
      do i = 1, n
        lambda_mod(i) = 1.0d0 / sqrt(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, inv_sqrt_a)
      
      call spd_triple_product(n, inv_sqrt_a, V, inv_sqrt_a, temp)
      
      eigenvectors = temp
#ifdef LNN_USE_DSYEVR
      if (SPD_EIG_PATH == SPD_EIG_DSYEVR) then
        allocate(isuppz(2*n))
        call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                    eigenvectors, n, isuppz, work, lwork, iwork, liwork, info)
        deallocate(isuppz)
      else
        call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
      end if
#else
      call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
#endif
      if (info /= 0) then
        write(*,*) 'LAPACK dsyev failed in lnn_spd_exp_log (temp eig, log) info =', info
        do i = 1, n*n
          out(i) = A(i)
        end do
        return
      end if
      
      do i = 1, n
        eigenvalues(i) = max(eigenvalues(i), EPS)
        lambda_mod(i) = log(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, temp)
      
      do i = 1, n
        lambda_mod(i) = sqrt(eigenvalues(i))
      end do
      call reconstruct_spd(n, eigenvectors, lambda_mod, sqrt_a)
      
      call spd_triple_product(n, sqrt_a, temp, sqrt_a, work_a)
    end if
    
    ! Copy result
    do j = 1, n
      do i = 1, n
        out((j-1)*n + i) = work_a(i, j)
      end do
    end do
    deallocate(work); deallocate(iwork)
#else
    ! Fallback without LAPACK
    write(*,*) 'Warning: lnn_spd_exp_log called without LAPACK; returning input A.'
    do i = 1, n*n
      out(i) = A(i)
    end do
#endif
  end subroutine lnn_spd_exp_log
  
  ! Parallel transport on SPD manifold
  subroutine lnn_spd_parallel_transport(n, A, B, V, out) bind(C, name='lnn_spd_parallel_transport')
    integer(c_int), value :: n
    real(c_double), intent(in) :: A(*), B(*), V(*)
    real(c_double), intent(out) :: out(*)
    
    ! PT_A→B(V) = B^{1/2} A^{-1/2} V A^{-1/2} B^{1/2}
    real(c_double) :: work_a(n, n), work_b(n, n), temp(n, n)
    real(c_double) :: eigenvalues_a(n), eigenvalues_b(n)
    real(c_double) :: eigenvectors_a(n, n), eigenvectors_b(n, n)
    real(c_double) :: sqrt_b(n, n), inv_sqrt_a(n, n)
    real(c_double), allocatable :: work(:)
    integer, allocatable :: iwork(:)
    ! Workspace query helpers
    real(c_double) :: work_q(1), vl, vu, abstol
    integer :: iwork_q(1), m
    integer, allocatable :: isuppz(:)
    real(c_double) :: lambda_mod(n)
    integer :: i, j, info, lwork, liwork
    character :: jobz, uplo
    
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
    ! Symmetrize inputs
    call lnn_read_env()
    do j = 1, n
      do i = 1, n
        work_a(i, j) = 0.5d0 * (A((j-1)*n + i) + A((i-1)*n + j))
        work_b(i, j) = 0.5d0 * (B((j-1)*n + i) + B((i-1)*n + j))
      end do
    end do
    
    jobz = 'V'
    uplo = 'U'
    ! Eigendecompose A
#ifdef LNN_USE_DSYEVR
    if (SPD_EIG_PATH == SPD_EIG_DSYEVR) then
      vl = 0.0d0; vu = 0.0d0; abstol = SPD_ABSTOL
      eigenvectors_a = work_a
      call dsyevr(jobz, 'A', uplo, n, eigenvectors_a, n, vl, vu, 0, 0, abstol, m, eigenvalues_a, &
                  eigenvectors_a, n, iwork_q, work_q, -1, iwork_q, -1, info)
      lwork = int(work_q(1)); liwork = iwork_q(1)
      allocate(work(lwork)); allocate(iwork(liwork)); allocate(isuppz(2*n))
      call dsyevr(jobz, 'A', uplo, n, eigenvectors_a, n, vl, vu, 0, 0, abstol, m, eigenvalues_a, &
                  eigenvectors_a, n, isuppz, work, lwork, iwork, liwork, info)
      deallocate(isuppz)
    else
      eigenvectors_a = work_a
      call dsyevd(jobz, uplo, n, eigenvectors_a, n, eigenvalues_a, work_q, -1, iwork_q, -1, info)
      lwork = int(work_q(1)); liwork = iwork_q(1)
      allocate(work(lwork)); allocate(iwork(liwork))
      call dsyevd(jobz, uplo, n, eigenvectors_a, n, eigenvalues_a, work, lwork, iwork, liwork, info)
    end if
#else
    eigenvectors_a = work_a
    call dsyevd(jobz, uplo, n, eigenvectors_a, n, eigenvalues_a, work_q, -1, iwork_q, -1, info)
    lwork = int(work_q(1)); liwork = iwork_q(1)
    allocate(work(lwork)); allocate(iwork(liwork))
    call dsyevd(jobz, uplo, n, eigenvectors_a, n, eigenvalues_a, work, lwork, iwork, liwork, info)
#endif
    if (info /= 0) then
      write(*,*) 'LAPACK dsyev failed in lnn_spd_parallel_transport (A eig) info =', info
      do i = 1, n*n
        out(i) = V(i)
      end do
      return
    end if
    
    do i = 1, n
      eigenvalues_a(i) = max(eigenvalues_a(i), EPS)
      lambda_mod(i) = 1.0d0 / sqrt(eigenvalues_a(i))
    end do
    call reconstruct_spd(n, eigenvectors_a, lambda_mod, inv_sqrt_a)
    
    ! Eigendecompose B
    eigenvectors_b = work_b
#ifdef LNN_USE_DSYEVR
    if (SPD_EIG_PATH == SPD_EIG_DSYEVR) then
      allocate(isuppz(2*n))
      call dsyevr(jobz, 'A', uplo, n, eigenvectors_b, n, vl, vu, 0, 0, abstol, m, eigenvalues_b, &
                  eigenvectors_b, n, isuppz, work, lwork, iwork, liwork, info)
      deallocate(isuppz)
    else
      call dsyevd(jobz, uplo, n, eigenvectors_b, n, eigenvalues_b, work, lwork, iwork, liwork, info)
    end if
#else
    call dsyevd(jobz, uplo, n, eigenvectors_b, n, eigenvalues_b, work, lwork, iwork, liwork, info)
#endif
    if (info /= 0) then
      write(*,*) 'LAPACK dsyev failed in lnn_spd_parallel_transport (B eig) info =', info
      do i = 1, n*n
        out(i) = V(i)
      end do
      return
    end if
    
    do i = 1, n
      eigenvalues_b(i) = max(eigenvalues_b(i), EPS)
      lambda_mod(i) = sqrt(eigenvalues_b(i))
    end do
    call reconstruct_spd(n, eigenvectors_b, lambda_mod, sqrt_b)
    
    ! Compute transport
    call spd_quad_product(n, sqrt_b, inv_sqrt_a, V, inv_sqrt_a, sqrt_b, temp)
    
    do j = 1, n
      do i = 1, n
        out((j-1)*n + i) = temp(i, j)
      end do
    end do
    deallocate(work); deallocate(iwork)
#else
    do i = 1, n*n
      out(i) = V(i)
    end do
#endif
  end subroutine lnn_spd_parallel_transport
  
  ! Reconstruct SPD matrix from eigendecomposition
  subroutine reconstruct_spd(n, Q, lambda, A)
    integer, intent(in) :: n
    real(c_double), intent(in) :: Q(n, n), lambda(n)
    real(c_double), intent(out) :: A(n, n)
    integer :: i, j, k
    
    ! A = Q * diag(lambda) * Q^T
    do i = 1, n
      do j = 1, n
        A(i, j) = 0.0d0
        do k = 1, n
          A(i, j) = A(i, j) + Q(i, k) * lambda(k) * Q(j, k)
        end do
      end do
    end do
  end subroutine reconstruct_spd
  
  ! Triple product: C = A * B * A
  subroutine spd_triple_product(n, A, B, C, out)
    integer, intent(in) :: n
    real(c_double), intent(in) :: A(n, n), B(*), C(n, n)
    real(c_double), intent(out) :: out(n, n)
    real(c_double) :: temp(n, n), work_b(n, n)
    integer :: i, j, k
    
    ! Convert B from flat to matrix
    do j = 1, n
      do i = 1, n
        work_b(i, j) = B((j-1)*n + i)
      end do
    end do
    
    ! temp = A * B
    do i = 1, n
      do j = 1, n
        temp(i, j) = 0.0d0
        do k = 1, n
          temp(i, j) = temp(i, j) + A(i, k) * work_b(k, j)
        end do
      end do
    end do
    
    ! out = temp * C
    do i = 1, n
      do j = 1, n
        out(i, j) = 0.0d0
        do k = 1, n
          out(i, j) = out(i, j) + temp(i, k) * C(k, j)
        end do
      end do
    end do
  end subroutine spd_triple_product
  
  ! Quad product: out = A * B * C * D * E
  subroutine spd_quad_product(n, A, B, C, D, E, out)
    integer, intent(in) :: n
    real(c_double), intent(in) :: A(n, n), B(n, n), C(*), D(n, n), E(n, n)
    real(c_double), intent(out) :: out(n, n)
    real(c_double) :: temp1(n, n), temp2(n, n), work_c(n, n)
    integer :: i, j, k
    
    do j = 1, n
      do i = 1, n
        work_c(i, j) = C((j-1)*n + i)
      end do
    end do
    
    ! temp1 = B * C
    do i = 1, n
      do j = 1, n
        temp1(i, j) = 0.0d0
        do k = 1, n
          temp1(i, j) = temp1(i, j) + B(i, k) * work_c(k, j)
        end do
      end do
    end do
    
    ! temp2 = temp1 * D
    do i = 1, n
      do j = 1, n
        temp2(i, j) = 0.0d0
        do k = 1, n
          temp2(i, j) = temp2(i, j) + temp1(i, k) * D(k, j)
        end do
      end do
    end do
    
    ! temp1 = A * temp2
    do i = 1, n
      do j = 1, n
        temp1(i, j) = 0.0d0
        do k = 1, n
          temp1(i, j) = temp1(i, j) + A(i, k) * temp2(k, j)
        end do
      end do
    end do
    
    ! out = temp1 * E
    do i = 1, n
      do j = 1, n
        out(i, j) = 0.0d0
        do k = 1, n
          out(i, j) = out(i, j) + temp1(i, k) * E(k, j)
        end do
      end do
    end do
  end subroutine spd_quad_product
  
  ! SPD geodesic distance
  subroutine lnn_spd_distance(n, A, B, dist) bind(C, name='lnn_spd_distance')
    integer(c_int), value :: n
    real(c_double), intent(in) :: A(*), B(*)
    real(c_double), intent(out) :: dist
    
    ! dist(A,B) = ||log(A^{-1/2} B A^{-1/2})||_F
    ! Requires LAPACK for eigendecomposition of SPD matrices
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
    real(c_double) :: work_a(n, n), work_b(n, n), inv_sqrt_a(n, n)
    real(c_double) :: eigenvalues(n), eigenvectors(n, n)
    real(c_double) :: lambda_mod(n)
    real(c_double) :: temp(n, n)
    real(c_double), allocatable :: work(:)
    integer, allocatable :: iwork(:)
    integer :: i, j, info, lwork, liwork
    character :: jobz, uplo
    ! Workspace query helpers
    real(c_double) :: work_q(1), vl, vu, abstol
    integer :: iwork_q(1), m
    integer, allocatable :: isuppz(:)

    ! Symmetrize A and B
    call lnn_read_env()
    do j = 1, n
      do i = 1, n
        work_a(i, j) = 0.5d0 * (A((j-1)*n + i) + A((i-1)*n + j))
        work_b(i, j) = 0.5d0 * (B((j-1)*n + i) + B((i-1)*n + j))
      end do
    end do

    jobz = 'V'; uplo = 'U'

    ! Eigendecompose A to get A^{-1/2}
    eigenvectors = work_a
#if defined(USE_LAPACK_MRRR) || defined(LNN_USE_DSYEVR)
    vl = 0.0d0; vu = 0.0d0; abstol = SPD_ABSTOL
    call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                eigenvectors, n, isuppz_q, work_q, -1, iwork_q, -1, info)
    lwork = int(work_q(1)); liwork = iwork_q(1)
    allocate(work(lwork)); allocate(iwork(liwork)); allocate(isuppz(2*n))
    call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                eigenvectors, n, isuppz, work, lwork, iwork, liwork, info)
    deallocate(isuppz)
#else
    call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work_q, -1, iwork_q, -1, info)
    lwork = int(work_q(1)); liwork = iwork_q(1)
    allocate(work(lwork)); allocate(iwork(liwork))
    call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
#endif
    if (info /= 0) then
      write(*,*) 'LAPACK dsyev failed in lnn_spd_distance (A eig) info =', info
      dist = 0.0d0
      return
    end if
    do i = 1, n
      eigenvalues(i) = max(eigenvalues(i), EPS)
      lambda_mod(i) = 1.0d0 / sqrt(eigenvalues(i))
    end do
    call reconstruct_spd(n, eigenvectors, lambda_mod, inv_sqrt_a)

    ! Compute P = A^{-1/2} B A^{-1/2}
    call spd_triple_product(n, inv_sqrt_a, B, inv_sqrt_a, temp)

    ! Eigendecompose P and accumulate squared logs
    eigenvectors = temp
#if defined(USE_LAPACK_MRRR) || defined(LNN_USE_DSYEVR)
    allocate(isuppz(2*n))
    call dsyevr(jobz, 'A', uplo, n, eigenvectors, n, vl, vu, 0, 0, abstol, m, eigenvalues, &
                eigenvectors, n, isuppz, work, lwork, iwork, liwork, info)
    deallocate(isuppz)
#else
    call dsyevd(jobz, uplo, n, eigenvectors, n, eigenvalues, work, lwork, iwork, liwork, info)
#endif
    if (info /= 0) then
      write(*,*) 'LAPACK dsyev failed in lnn_spd_distance (P eig) info =', info
      dist = 0.0d0
      return
    end if

    dist = 0.0d0
    do i = 1, n
      eigenvalues(i) = max(eigenvalues(i), EPS)
      dist = dist + log(eigenvalues(i))**2
    end do
    dist = sqrt(dist)
    deallocate(work); deallocate(iwork)
#else
    dist = 0.0d0
#endif
  end subroutine lnn_spd_distance
  
  ! ============================================================================
  ! HYPERBOLIC (POINCARÉ BALL) OPERATIONS
  ! ============================================================================
  
  ! Möbius addition in Poincaré ball
  subroutine lnn_mobius_add(n, x, y, out) bind(C, name='lnn_mobius_add')
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(*), y(*)
    real(c_double), intent(out) :: out(*)
    
    real(c_double) :: norm_x_sq, norm_y_sq, xy_dot, denom
    integer :: i
    
    norm_x_sq = 0.0d0
    norm_y_sq = 0.0d0
    xy_dot = 0.0d0
    
    do i = 1, n
      norm_x_sq = norm_x_sq + x(i)**2
      norm_y_sq = norm_y_sq + y(i)**2
      xy_dot = xy_dot + x(i) * y(i)
    end do
    
    denom = 1.0d0 + 2.0d0*xy_dot + norm_x_sq*norm_y_sq
    
    do i = 1, n
      out(i) = ((1.0d0 + 2.0d0*xy_dot + norm_y_sq)*x(i) + (1.0d0 - norm_x_sq)*y(i)) / denom
    end do
  end subroutine lnn_mobius_add
  
  ! Poincaré exponential map
  subroutine lnn_poincare_exp(n, x, v, out) bind(C, name='lnn_poincare_exp')
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(*), v(*)
    real(c_double), intent(out) :: out(*)
    
    real(c_double) :: norm_x_sq, norm_v, lambda_x, factor
    integer :: i
    
    norm_x_sq = 0.0d0
    norm_v = 0.0d0
    
    do i = 1, n
      norm_x_sq = norm_x_sq + x(i)**2
      norm_v = norm_v + v(i)**2
    end do
    norm_v = sqrt(norm_v)
    
    lambda_x = 2.0d0 / (1.0d0 - norm_x_sq)
    
    if (norm_v < EPS) then
      do i = 1, n
        out(i) = x(i)
      end do
    else
      factor = tanh(lambda_x * norm_v / 2.0d0) / norm_v
      call lnn_mobius_add(n, x, v, out)  ! Simplified
    end if
  end subroutine lnn_poincare_exp
  
  ! ============================================================================
  ! HELPER FUNCTIONS
  ! ============================================================================
  
  subroutine naive_matmul(m, n, k, A, B, C)
    integer, intent(in) :: m, n, k
    real(c_double), intent(in) :: A(k, m), B(n, k)
    real(c_double), intent(out) :: C(n, m)
    integer :: i, j, p
    real(c_double) :: sum_val
    
    do i = 1, m
      do j = 1, n
        sum_val = 0.0d0
        do p = 1, k
          sum_val = sum_val + A(p, i) * B(j, p)
        end do
        C(j, i) = sum_val
      end do
    end do
  end subroutine naive_matmul

end module tensor_kernels
