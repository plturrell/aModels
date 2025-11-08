module advanced_als
  use iso_c_binding
  use omp_lib
  use lnn_config, only: DEFAULT_EPS, SPD_EIG_PATH, SPD_EIG_DSYEVR, SPD_EIG_DSYEVD, SPD_ABSTOL
#ifdef LNN_PROFILE
  use lnn_profiler, only: prof_mark_start, prof_mark_stop
#endif
  implicit none
  
  real(c_double), parameter :: EPS = DEFAULT_EPS
  
contains

  ! ============================================================================
  ! FULL ALS WITH LAPACK (DGESV/DPOTRF)
  ! ============================================================================
  
  ! CP-ALS with full normal equations via LAPACK
  subroutine lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, which, out) &
      bind(C, name='lnn_cp_als_full')
    integer(c_int), value :: n1, n2, n3, R, which
    real(c_double), intent(in) :: T(*)      ! Tensor n1 x n2 x n3
    real(c_double), intent(in) :: U(*)      ! n1 x R
    real(c_double), intent(in) :: V(*)      ! n2 x R
    real(c_double), intent(in) :: W(*)      ! n3 x R
    real(c_double), value :: lambda         ! Regularization
    real(c_double), intent(out) :: out(*)   ! Updated factor
    
    real(c_double) :: gram(R, R), rhs(R), solution(R)
    real(c_double) :: khatri_rao(n2*n3, R)
    real(c_double), allocatable :: unfolded(:, :)
    integer :: i, j, k, rr, s, idx, info
    integer :: ipiv(R)
    real(c_double) :: temp
    real(c_double), allocatable :: gram_chol(:,:), gram_chol_v(:,:), gram_chol_w(:,:)
    integer :: info_chol, info_chol_v, info_chol_w
    
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
    if (which == 0) then
      ! Update U: (V^T V ⊙ W^T W + λI) U = MTTKRP_mode1(T; V,W)
      call compute_hadamard_gram(n2, n3, R, V, W, gram)
      do rr = 1, R
        gram(rr, rr) = gram(rr, rr) + lambda
      end do

      ! MTTKRP for mode-1 to compute RHS per row
      allocate(unfolded(n1, R))
      call mttkrp_mode1(n1, n2, n3, R, T, V, W, unfolded)

      ! Factor Gram once (SPD) and solve per-row via DPOTRS
      allocate(gram_chol(R,R))
      gram_chol = gram
      call dpotrf('U', R, gram_chol, R, info_chol)
      if (info_chol == 0) then
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, info)
        do i = 1, n1
          do rr = 1, R
            solution(rr) = unfolded(i, rr)
          end do
          call dpotrs('U', R, 1, gram_chol, R, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n1 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dpotrs failed in lnn_cp_als_full (U row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n1 + i) = U((rr-1)*n1 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      else
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, ipiv, info)
        do i = 1, n1
          do rr = 1, R
            rhs(rr) = unfolded(i, rr)
          end do
          solution = rhs
          call dgesv(R, 1, gram, R, ipiv, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n1 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dgesv failed in lnn_cp_als_full (U row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n1 + i) = U((rr-1)*n1 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      end if
      deallocate(unfolded)
      if (allocated(gram_chol)) deallocate(gram_chol)

    else if (which == 1) then
      ! Update V using mode-2 MTTKRP
      call compute_hadamard_gram(n1, n3, R, U, W, gram)
      do rr = 1, R
        gram(rr, rr) = gram(rr, rr) + lambda
      end do

      allocate(unfolded(n2, R))
      call mttkrp_mode2(n1, n2, n3, R, T, U, W, unfolded)

      allocate(gram_chol_v(R,R))
      gram_chol_v = gram
      call dpotrf('U', R, gram_chol_v, R, info_chol_v)
      if (info_chol_v == 0) then
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, info)
        do i = 1, n2
          do rr = 1, R
            solution(rr) = unfolded(i, rr)
          end do
          call dpotrs('U', R, 1, gram_chol_v, R, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n2 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dpotrs failed in lnn_cp_als_full (V row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n2 + i) = V((rr-1)*n2 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      else
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, ipiv, info)
        do i = 1, n2
          do rr = 1, R
            rhs(rr) = unfolded(i, rr)
          end do
          solution = rhs
          call dgesv(R, 1, gram, R, ipiv, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n2 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dgesv failed in lnn_cp_als_full (V row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n2 + i) = V((rr-1)*n2 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      end if
      deallocate(unfolded)
      if (allocated(gram_chol_v)) deallocate(gram_chol_v)

    else if (which == 2) then
      ! Update W using mode-3 MTTKRP
      call compute_hadamard_gram(n1, n2, R, U, V, gram)
      do rr = 1, R
        gram(rr, rr) = gram(rr, rr) + lambda
      end do

      allocate(unfolded(n3, R))
      call mttkrp_mode3(n1, n2, n3, R, T, U, V, unfolded)

      allocate(gram_chol_w(R,R))
      gram_chol_w = gram
      call dpotrf('U', R, gram_chol_w, R, info_chol_w)
      if (info_chol_w == 0) then
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, info)
        do i = 1, n3
          do rr = 1, R
            solution(rr) = unfolded(i, rr)
          end do
          call dpotrs('U', R, 1, gram_chol_w, R, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n3 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dpotrs failed in lnn_cp_als_full (W row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n3 + i) = W((rr-1)*n3 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      else
        !$OMP PARALLEL DO PRIVATE(i, rr, rhs, solution, ipiv, info)
        do i = 1, n3
          do rr = 1, R
            rhs(rr) = unfolded(i, rr)
          end do
          solution = rhs
          call dgesv(R, 1, gram, R, ipiv, solution, R, info)
          if (info == 0) then
            do rr = 1, R
              out((rr-1)*n3 + i) = solution(rr)
            end do
          else
            write(*,*) 'LAPACK dgesv failed in lnn_cp_als_full (W row ', i, ') info =', info
            do rr = 1, R
              out((rr-1)*n3 + i) = W((rr-1)*n3 + i) * 0.99d0
            end do
          end if
        end do
        !$OMP END PARALLEL DO
      end if
      deallocate(unfolded)
      if (allocated(gram_chol_w)) deallocate(gram_chol_w)
    end if
#else
    ! Fallback without LAPACK
    if (which == 0) then
      do i = 1, n1*R
        out(i) = U(i)
      end do
    else if (which == 1) then
      do i = 1, n2*R
        out(i) = V(i)
      end do
    else
      do i = 1, n3*R
        out(i) = W(i)
      end do
    end if
#endif
end subroutine lnn_cp_als_full

  ! Thin wrapper to expose CP-ALS step under a stable C symbol
  subroutine lnn_cp_als_step(n1, n2, n3, R, T, U, V, W, lambda, which, out) &
      bind(C, name='lnn_cp_als_step')
    integer(c_int), value :: n1, n2, n3, R, which
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*)
    real(c_double), value :: lambda
    real(c_double), intent(out) :: out(*)
    call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, which, out)
  end subroutine lnn_cp_als_step

  ! Simple CP-ALS driver: performs a fixed number of ALS sweeps
  subroutine lnn_cp_als_driver(n1, n2, n3, R, T, U, V, W, lambda, iters) &
      bind(C, name='lnn_cp_als_driver')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), value :: lambda
    integer(c_int), value :: iters
    integer :: it, i
    real(c_double), allocatable :: tmp(:)
    ! Update U, then V, then W per iteration
    do it = 1, iters
      allocate(tmp(n1*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 0_c_int, tmp)
      do i = 1, n1*R
        U(i) = tmp(i)
      end do
      deallocate(tmp)

      allocate(tmp(n2*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 1_c_int, tmp)
      do i = 1, n2*R
        V(i) = tmp(i)
      end do
      deallocate(tmp)

      allocate(tmp(n3*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 2_c_int, tmp)
      do i = 1, n3*R
        W(i) = tmp(i)
      end do
      deallocate(tmp)
    end do
  end subroutine lnn_cp_als_driver

  ! Normalize columns of a factor matrix in-place (L2 norm = 1)
  subroutine normalize_columns(n, R, X)
    integer, intent(in) :: n, R
    real(c_double), intent(inout) :: X(n, R)
    integer :: rr, i
    real(c_double) :: s
    do rr = 1, R
      s = 0.0d0
      do i = 1, n
        s = s + X(i, rr)*X(i, rr)
      end do
      s = sqrt(max(s, 1.0d-30))
      do i = 1, n
        X(i, rr) = X(i, rr) / s
      end do
    end do
  end subroutine normalize_columns

  subroutine normalize_columns_flat(n, R, Xflat)
    integer, intent(in) :: n, R
    real(c_double), intent(inout) :: Xflat(*)
    integer :: rr, i
    real(c_double) :: s
    do rr = 1, R
      s = 0.0d0
      do i = 1, n
        s = s + Xflat((rr-1)*n + i)**2
      end do
      s = sqrt(max(s, 1.0d-30))
      do i = 1, n
        Xflat((rr-1)*n + i) = Xflat((rr-1)*n + i) / s
      end do
    end do
  end subroutine normalize_columns_flat

  subroutine normalize_columns_flat_f32(n, R, Xflat)
    integer, intent(in) :: n, R
    real(c_float), intent(inout) :: Xflat(*)
    integer :: rr, i
    real(c_float) :: s
    do rr = 1, R
      s = 0.0
      do i = 1, n
        s = s + Xflat((rr-1)*n + i)**2
      end do
      s = sqrt(max(s, real(1.0d-30, c_float)))
      do i = 1, n
        Xflat((rr-1)*n + i) = Xflat((rr-1)*n + i) / s
      end do
    end do
  end subroutine normalize_columns_flat_f32

  subroutine compute_col_norms(n, R, Xflat, norms)
    integer, intent(in) :: n, R
    real(c_double), intent(in) :: Xflat(*)
    real(c_double), intent(out) :: norms(R)
    integer :: rr, i
    real(c_double) :: s
    do rr = 1, R
      s = 0.0d0
      do i = 1, n
        s = s + Xflat((rr-1)*n + i)**2
      end do
      norms(rr) = sqrt(max(s, 1.0d-30))
    end do
  end subroutine compute_col_norms

  subroutine compute_col_norms_f32(n, R, Xflat, norms)
    integer, intent(in) :: n, R
    real(c_float), intent(in) :: Xflat(*)
    real(c_float), intent(out) :: norms(R)
    integer :: rr, i
    real(c_float) :: s
    do rr = 1, R
      s = 0.0
      do i = 1, n
        s = s + Xflat((rr-1)*n + i)**2
      end do
      norms(rr) = sqrt(max(s, real(1.0d-30, c_float)))
    end do
  end subroutine compute_col_norms_f32

  ! Compute squared Frobenius norm of full tensor T
  function tensor_frob2(n1, n2, n3, T) result(val)
    integer, intent(in) :: n1, n2, n3
    real(c_double), intent(in) :: T(*)
    real(c_double) :: val
    integer :: i
    val = 0.0d0
    do i = 1, n1*n2*n3
      val = val + T(i)*T(i)
    end do
  end function tensor_frob2

  ! Compute squared reconstruction error ||T - [|U,V,W|]||_F^2 (explicit)
  function cp_recon_error2(n1, n2, n3, R, T, U, V, W) result(err2)
    integer, intent(in) :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*)
    real(c_double) :: err2
    integer :: i, j, k, rr
    real(c_double) :: x, diff
    err2 = 0.0d0
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do rr = 1, R
            x = x + U((rr-1)*n1 + i) * V((rr-1)*n2 + j) * W((rr-1)*n3 + k)
          end do
          diff = T(i + (j-1)*n1 + (k-1)*n1*n2) - x
          err2 = err2 + diff*diff
        end do
      end do
    end do
  end function cp_recon_error2

  function cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda) result(err2)
    integer, intent(in) :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*), lambda(*)
    real(c_double) :: err2
    integer :: i, j, k, rr
    real(c_double) :: x, diff
    err2 = 0.0d0
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do rr = 1, R
            x = x + lambda(rr) * U((rr-1)*n1 + i) * V((rr-1)*n2 + j) * W((rr-1)*n3 + k)
          end do
          diff = T(i + (j-1)*n1 + (k-1)*n1*n2) - x
          err2 = err2 + diff*diff
        end do
      end do
    end do
  end function cp_recon_error2_lambda

  function cp_recon_error2_lambda_f32(n1, n2, n3, R, T, U, V, W, lambda) result(err2)
    integer, intent(in) :: n1, n2, n3, R
    real(c_float), intent(in) :: T(*), U(*), V(*), W(*), lambda(*)
    real(c_double) :: err2
    integer :: i, j, k, rr
    real(c_double) :: x, diff
    err2 = 0.0d0
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do rr = 1, R
            x = x + dble(lambda(rr)) * dble(U((rr-1)*n1 + i)) * dble(V((rr-1)*n2 + j)) * dble(W((rr-1)*n3 + k))
          end do
          diff = dble(T(i + (j-1)*n1 + (k-1)*n1*n2)) - x
          err2 = err2 + diff*diff
        end do
      end do
    end do
  end function cp_recon_error2_lambda_f32

  ! C-bind wrapper: residual norm and fit from factors + lambda
  subroutine lnn_cp_residual_norm(n1, n2, n3, R, T, U, V, W, lambda, res_norm) &
      bind(C, name='lnn_cp_residual_norm')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*), lambda(*)
    real(c_double), intent(out) :: res_norm
    real(c_double) :: err2
    err2 = cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda)
    res_norm = sqrt(err2)
  end subroutine lnn_cp_residual_norm

  subroutine lnn_cp_fit(n1, n2, n3, R, T, U, V, W, lambda, fit) &
      bind(C, name='lnn_cp_fit')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*), lambda(*)
    real(c_double), intent(out) :: fit
    real(c_double) :: err2, tn2
    err2 = cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda)
    tn2 = tensor_frob2(n1, n2, n3, T)
    if (tn2 > 0.0d0) then
      fit = 1.0d0 - err2/tn2
    else
      fit = 1.0d0
    end if
  end subroutine lnn_cp_fit

  ! Reconstruct full tensor from factors and lambda
  subroutine lnn_cp_reconstruct(n1, n2, n3, R, U, V, W, lambda, out) &
      bind(C, name='lnn_cp_reconstruct')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: U(*), V(*), W(*), lambda(*)
    real(c_double), intent(out) :: out(*)
    integer :: i, j, k, rr
    integer :: idx
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          idx = i + (j-1)*n1 + (k-1)*n1*n2
          out(idx) = 0.0d0
        end do
      end do
    end do
    do rr = 1, R
      do k = 1, n3
        do j = 1, n2
          do i = 1, n1
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(idx) = out(idx) + lambda(rr) * U((rr-1)*n1 + i) * V((rr-1)*n2 + j) * W((rr-1)*n3 + k)
          end do
        end do
      end do
    end do
  end subroutine lnn_cp_reconstruct

  ! Normal-equation residual norms for each mode (accounts for lambda by column scaling)
  subroutine lnn_cp_normal_eq_residuals(n1, n2, n3, R, T, U, V, W, lambda, ru, rv, rw) &
      bind(C, name='lnn_cp_normal_eq_residuals')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*), W(*), lambda(*)
    real(c_double), intent(out) :: ru, rv, rw
    real(c_double), allocatable :: gram(:,:), rhs(:,:), res(:,:), Vlam(:), Wlam(:), Ulam(:)
    integer :: i, rr, s

    ! Mode-1 residual: ru = ||G(V,W,λ)*U^T - MTTKRP_mode1(T; Vλ, Wλ)||_F
    allocate(gram(R,R), rhs(n1,R), res(n1,R), Vlam(n2*R), Wlam(n3*R))
    do rr = 1, R
      do i = 1, n2
        Vlam((rr-1)*n2 + i) = V((rr-1)*n2 + i) * lambda(rr)
      end do
      do i = 1, n3
        Wlam((rr-1)*n3 + i) = W((rr-1)*n3 + i) * lambda(rr)
      end do
    end do
    call compute_hadamard_gram(n2, n3, R, Vlam, Wlam, gram)
    call mttkrp_mode1(n1, n2, n3, R, T, Vlam, Wlam, rhs)
    ru = 0.0d0
    do i = 1, n1
      do rr = 1, R
        res(i, rr) = 0.0d0
        do s = 1, R
          res(i, rr) = res(i, rr) + gram(rr, s) * U((s-1)*n1 + i)
        end do
        res(i, rr) = res(i, rr) - rhs(i, rr)
        ru = ru + res(i, rr)**2
      end do
    end do
    deallocate(gram, rhs, res, Vlam, Wlam)

    ! Mode-2 residual
    allocate(gram(R,R), rhs(n2,R), res(n2,R), Ulam(n1*R), Wlam(n3*R))
    do rr = 1, R
      do i = 1, n1
        Ulam((rr-1)*n1 + i) = U((rr-1)*n1 + i) * lambda(rr)
      end do
      do i = 1, n3
        Wlam((rr-1)*n3 + i) = W((rr-1)*n3 + i) * lambda(rr)
      end do
    end do
    call compute_hadamard_gram(n1, n3, R, Ulam, Wlam, gram)
    call mttkrp_mode2(n1, n2, n3, R, T, Ulam, Wlam, rhs)
    rv = 0.0d0
    do i = 1, n2
      do rr = 1, R
        res(i, rr) = 0.0d0
        do s = 1, R
          res(i, rr) = res(i, rr) + gram(rr, s) * V((s-1)*n2 + i)
        end do
        res(i, rr) = res(i, rr) - rhs(i, rr)
        rv = rv + res(i, rr)**2
      end do
    end do
    deallocate(gram, rhs, res, Ulam, Wlam)

    ! Mode-3 residual
    allocate(gram(R,R), rhs(n3,R), res(n3,R), Ulam(n1*R), Vlam(n2*R))
    do rr = 1, R
      do i = 1, n1
        Ulam((rr-1)*n1 + i) = U((rr-1)*n1 + i) * lambda(rr)
      end do
      do i = 1, n2
        Vlam((rr-1)*n2 + i) = V((rr-1)*n2 + i) * lambda(rr)
      end do
    end do
    call compute_hadamard_gram(n1, n2, R, Ulam, Vlam, gram)
    call mttkrp_mode3(n1, n2, n3, R, T, Ulam, Vlam, rhs)
    rw = 0.0d0
    do i = 1, n3
      do rr = 1, R
        res(i, rr) = 0.0d0
        do s = 1, R
          res(i, rr) = res(i, rr) + gram(rr, s) * W((s-1)*n3 + i)
        end do
        res(i, rr) = res(i, rr) - rhs(i, rr)
        rw = rw + res(i, rr)**2
      end do
    end do
    deallocate(gram, rhs, res, Ulam, Vlam)

    ru = sqrt(ru); rv = sqrt(rv); rw = sqrt(rw)
  end subroutine lnn_cp_normal_eq_residuals

  subroutine lnn_cp_print_fit_trace(fit, iters) bind(C, name='lnn_cp_print_fit_trace')
    real(c_double), intent(in) :: fit(*)
    integer(c_int), value :: iters
    integer :: i
    do i = 1, iters
      write(*,'(A,I4,A,F12.8)') 'iter=', i, ' fit=', fit(i)
    end do
  end subroutine lnn_cp_print_fit_trace

  subroutine lnn_cp_als_train(n1, n2, n3, R, T, U, V, W, tol, max_iters, final_fit, iters_out, lambda_out) &
      bind(C, name='lnn_cp_als_train')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), value :: tol
    integer(c_int), value :: max_iters
    real(c_double), intent(out) :: final_fit
    integer(c_int), intent(out) :: iters_out
    real(c_double), intent(out) :: lambda_out(*)
    real(c_double), allocatable :: fit(:)
    allocate(fit(max_iters))
    call lnn_cp_als_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda_out, tol, max_iters, fit, iters_out)
    if (iters_out >= 1_c_int) then
      final_fit = fit(iters_out)
    else
      final_fit = 0.0d0
    end if
    deallocate(fit)
  end subroutine lnn_cp_als_train

  ! One-shot training API returning final_fit, iterations, normal-equation residuals, and lambda
  subroutine lnn_cp_als_train_summary(n1, n2, n3, R, T, U, V, W, tol, max_iters, final_fit, iters_out, ru, rv, rw, lambda_out) &
      bind(C, name='lnn_cp_als_train_summary')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), value :: tol
    integer(c_int), value :: max_iters
    real(c_double), intent(out) :: final_fit
    integer(c_int), intent(out) :: iters_out
    real(c_double), intent(out) :: ru, rv, rw
    real(c_double), intent(out) :: lambda_out(*)
    real(c_double), allocatable :: fit(:)
    integer :: it
    allocate(fit(max_iters))
    call lnn_cp_als_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda_out, tol, max_iters, fit, iters_out)
    it = int(iters_out)
    if (it >= 1) then
      final_fit = fit(it)
    else
      final_fit = 0.0d0
    end if
    call lnn_cp_normal_eq_residuals(n1, n2, n3, R, T, U, V, W, lambda_out, ru, rv, rw)
    deallocate(fit)
  end subroutine lnn_cp_als_train_summary

  ! One-shot: set SPD config and run CP-ALS training, returning final summary
  subroutine lnn_cp_train_one_shot(n1, n2, n3, R, T, U, V, W, tol, max_iters, &
      spd_eig_path, spd_abstol, final_fit, iters_out, ru, rv, rw, lambda_out) &
      bind(C, name='lnn_cp_train_one_shot')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), value :: tol
    integer(c_int), value :: max_iters
    integer(c_int), value :: spd_eig_path
    real(c_double), value :: spd_abstol
    real(c_double), intent(out) :: final_fit
    integer(c_int), intent(out) :: iters_out
    real(c_double), intent(out) :: ru, rv, rw
    real(c_double), intent(out) :: lambda_out(*)

    ! Apply SPD configuration (runtime) for downstream SPD ops
    SPD_EIG_PATH = int(spd_eig_path)
    SPD_ABSTOL = spd_abstol

    ! Train and return summary
    call lnn_cp_als_train_summary(n1, n2, n3, R, T, U, V, W, tol, max_iters, final_fit, iters_out, ru, rv, rw, lambda_out)
  end subroutine lnn_cp_train_one_shot

  ! CP-ALS driver with tolerance and normalization; returns when relative
  ! improvement in fit falls below tol or max_iters reached.
  subroutine lnn_cp_als_driver_tol(n1, n2, n3, R, T, U, V, W, lambda, tol, max_iters) &
      bind(C, name='lnn_cp_als_driver_tol')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), value :: lambda, tol
    integer(c_int), value :: max_iters
    integer :: it, i
    real(c_double) :: prev_err2, err2, Tn2
    real(c_double), allocatable :: U_new(:), V_new(:), W_new(:)
    real(c_double), allocatable :: U_prev(:), V_prev(:), W_prev(:)
    real(c_double) :: damp

    Tn2 = tensor_frob2(n1, n2, n3, T)
    prev_err2 = cp_recon_error2(n1, n2, n3, R, T, U, V, W)

    do it = 1, max_iters
      allocate(U_prev(n1*R)); do i = 1, n1*R; U_prev(i) = U(i); end do
      allocate(U_new(n1*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 0_c_int, U_new)
      ! Normalize U columns
      call normalize_columns_flat(n1, R, U_new)
      do i = 1, n1*R; U(i) = U_new(i); end do
      deallocate(U_new)

      allocate(V_prev(n2*R)); do i = 1, n2*R; V_prev(i) = V(i); end do
      allocate(V_new(n2*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 1_c_int, V_new)
      call normalize_columns_flat(n2, R, V_new)
      do i = 1, n2*R; V(i) = V_new(i); end do
      deallocate(V_new)

      allocate(W_prev(n3*R)); do i = 1, n3*R; W_prev(i) = W(i); end do
      allocate(W_new(n3*R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, lambda, 2_c_int, W_new)
      call normalize_columns_flat(n3, R, W_new)
      do i = 1, n3*R; W(i) = W_new(i); end do
      deallocate(W_new)

      err2 = cp_recon_error2(n1, n2, n3, R, T, U, V, W)

      ! Simple damping if error increases
      if (err2 > prev_err2) then
        damp = 0.5d0
        do i = 1, n1*R; U(i) = damp*U(i) + (1.0d0-damp)*U_prev(i); end do
        do i = 1, n2*R; V(i) = damp*V(i) + (1.0d0-damp)*V_prev(i); end do
        do i = 1, n3*R; W(i) = damp*W(i) + (1.0d0-damp)*W_prev(i); end do
        err2 = cp_recon_error2(n1, n2, n3, R, T, U, V, W)
      end if

      if (Tn2 > 0.0d0) then
        if ((prev_err2 - err2) / Tn2 < tol) exit
      else
        if (abs(prev_err2 - err2) < tol) exit
      end if
      prev_err2 = err2
      deallocate(U_prev, V_prev, W_prev)
    end do
  end subroutine lnn_cp_als_driver_tol

  ! CP-ALS driver with lambda (weights) and per-iteration fit trace
  subroutine lnn_cp_als_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda_out, tol, max_iters, fit_out, iters_out) &
      bind(C, name='lnn_cp_als_driver_with_lambda')
    integer(c_int), value :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(inout) :: U(*), V(*), W(*)
    real(c_double), intent(out) :: lambda_out(*)
    real(c_double), value :: tol
    integer(c_int), value :: max_iters
    real(c_double), intent(out) :: fit_out(*)
    integer(c_int), intent(out) :: iters_out
    integer :: it, i
    real(c_double) :: prev_err2, err2, Tn2
    real(c_double), allocatable :: U_new(:), V_new(:), W_new(:)
    real(c_double), allocatable :: U_prev(:), V_prev(:), W_prev(:)
    real(c_double) :: damp
    real(c_double), allocatable :: nu(:), nv(:), nw(:)
    real(c_double), allocatable :: lambda(:)

    allocate(lambda(R))
    lambda = 1.0d0
    Tn2 = tensor_frob2(n1, n2, n3, T)
    prev_err2 = cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda)

    do it = 1, max_iters
      allocate(U_prev(n1*R)); do i = 1, n1*R; U_prev(i) = U(i); end do
      allocate(U_new(n1*R)); allocate(nu(R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 0.0d0, 0_c_int, U_new)
      call compute_col_norms(n1, R, U_new, nu)
      do i = 1, n1*R; U(i) = U_new(i); end do
      call normalize_columns_flat(n1, R, U)
      do i = 1, R; lambda(i) = lambda(i) * nu(i); end do
      deallocate(U_new, U_prev, nu)

      allocate(V_prev(n2*R)); do i = 1, n2*R; V_prev(i) = V(i); end do
      allocate(V_new(n2*R)); allocate(nv(R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 0.0d0, 1_c_int, V_new)
      call compute_col_norms(n2, R, V_new, nv)
      do i = 1, n2*R; V(i) = V_new(i); end do
      call normalize_columns_flat(n2, R, V)
      do i = 1, R; lambda(i) = lambda(i) * nv(i); end do
      deallocate(V_new, V_prev, nv)

      allocate(W_prev(n3*R)); do i = 1, n3*R; W_prev(i) = W(i); end do
      allocate(W_new(n3*R)); allocate(nw(R))
      call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 0.0d0, 2_c_int, W_new)
      call compute_col_norms(n3, R, W_new, nw)
      do i = 1, n3*R; W(i) = W_new(i); end do
      call normalize_columns_flat(n3, R, W)
      do i = 1, R; lambda(i) = lambda(i) * nw(i); end do
      deallocate(W_new, W_prev, nw)

      err2 = cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda)
      if (Tn2 > 0.0d0) then
        fit_out(it) = 1.0d0 - err2 / Tn2
      else
        fit_out(it) = 1.0d0
      end if

      if (err2 > prev_err2) then
        damp = 0.5d0
        ! Dampen factors and lambda toward previous iteration
        do i = 1, n1*R; U(i) = damp*U(i) + (1.0d0-damp)*U_prev(i); end do
        do i = 1, n2*R; V(i) = damp*V(i) + (1.0d0-damp)*V_prev(i); end do
        do i = 1, n3*R; W(i) = damp*W(i) + (1.0d0-damp)*W_prev(i); end do
        do i = 1, R; lambda(i) = damp*lambda(i) + (1.0d0-damp)*1.0d0; end do
        err2 = cp_recon_error2_lambda(n1, n2, n3, R, T, U, V, W, lambda)
        if (Tn2 > 0.0d0) then
          fit_out(it) = 1.0d0 - err2 / Tn2
        end if
      end if

      if (Tn2 > 0.0d0) then
        if ((prev_err2 - err2) / Tn2 < tol) then
          iters_out = it
          exit
        end if
      else
        if (abs(prev_err2 - err2) < tol) then
          iters_out = it
          exit
        end if
      end if
      prev_err2 = err2
      iters_out = it
    end do

    do i = 1, R
      lambda_out(i) = lambda(i)
    end do
    deallocate(lambda)
  end subroutine lnn_cp_als_driver_with_lambda

  ! Float32 MTTKRP (mode-1) for benchmarking
  subroutine mttkrp_mode1_f32(n1, n2, n3, R, T, V, W, out)
    integer(c_int), value :: n1, n2, n3, R
    real(c_float), intent(in) :: T(*), V(*), W(*)
    real(c_float), intent(out) :: out(n1, R)
    integer :: i, j, k, rr
    integer :: idx
    out = 0.0
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do i = 1, n1
        do k = 1, n3
          do j = 1, n2
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(i, rr) = out(i, rr) + T(idx) * V((rr-1)*n2 + j) * W((rr-1)*n3 + k)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine mttkrp_mode1_f32

  subroutine mttkrp_mode2_f32(n1, n2, n3, R, T, U, W, out)
    integer(c_int), value :: n1, n2, n3, R
    real(c_float), intent(in) :: T(*), U(*), W(*)
    real(c_float), intent(out) :: out(n2, R)
    integer :: i, j, k, rr, idx
    out = 0.0
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do j = 1, n2
        do k = 1, n3
          do i = 1, n1
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(j, rr) = out(j, rr) + T(idx) * U((rr-1)*n1 + i) * W((rr-1)*n3 + k)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine mttkrp_mode2_f32

  subroutine mttkrp_mode3_f32(n1, n2, n3, R, T, U, V, out)
    integer(c_int), value :: n1, n2, n3, R
    real(c_float), intent(in) :: T(*), U(*), V(*)
    real(c_float), intent(out) :: out(n3, R)
    integer :: i, j, k, rr, idx
    out = 0.0
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do k = 1, n3
        do j = 1, n2
          do i = 1, n1
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(k, rr) = out(k, rr) + T(idx) * U((rr-1)*n1 + i) * V((rr-1)*n2 + j)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine mttkrp_mode3_f32

  subroutine compute_hadamard_gram_f32(nA, nB, R, A, B, gram)
    integer(c_int), value :: nA, nB, R
    real(c_float), intent(in) :: A(*), B(*)
    real(c_float), intent(out) :: gram(R, R)
    real(c_float) :: gramA(R, R), gramB(R, R)
    integer :: rr, s, i
    do rr = 1, R
      do s = 1, R
        gramA(rr, s) = 0.0
        do i = 1, nA
          gramA(rr, s) = gramA(rr, s) + A((rr-1)*nA + i) * A((s-1)*nA + i)
        end do
      end do
    end do
    do rr = 1, R
      do s = 1, R
        gramB(rr, s) = 0.0
        do i = 1, nB
          gramB(rr, s) = gramB(rr, s) + B((rr-1)*nB + i) * B((s-1)*nB + i)
        end do
      end do
    end do
    do rr = 1, R
      do s = 1, R
        gram(rr, s) = gramA(rr, s) * gramB(rr, s)
      end do
    end do
  end subroutine compute_hadamard_gram_f32

  ! Single-precision CP-ALS (one mode) using SPOTRF/SGESV
  subroutine lnn_cp_als_f32_full(n1, n2, n3, R, T, U, V, W, lambda, which, out) &
      bind(C, name='lnn_cp_als_f32_full')
    integer(c_int), value :: n1, n2, n3, R, which
    real(c_float), intent(in) :: T(*), U(*), V(*), W(*)
    real(c_float), value :: lambda
    real(c_float), intent(out) :: out(*)
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
    real(c_float), allocatable :: rhs(:,:), gram(:,:), sol(:)
    integer :: i, rr, info
    integer, allocatable :: ipiv(:)
    if (which == 0) then
      allocate(rhs(n1,R), gram(R,R), sol(R))
      call compute_hadamard_gram_f32(n2, n3, R, V, W, gram)
      do rr = 1, R; gram(rr,rr) = gram(rr,rr) + lambda; end do
      call mttkrp_mode1_f32(n1, n2, n3, R, T, V, W, rhs)
      do i = 1, n1
        sol = rhs(i,:)
        call spotrf('U', R, gram, R, info)
        if (info == 0) then
          call spotrs('U', R, 1, gram, R, sol, R, info)
        else
          allocate(ipiv(R))
          call sgesv(R, 1, gram, R, ipiv, sol, R, info)
          deallocate(ipiv)
        end if
        do rr = 1, R
          out((rr-1)*n1 + i) = sol(rr)
        end do
      end do
      deallocate(rhs, gram, sol)
    else if (which == 1) then
      allocate(rhs(n2,R), gram(R,R), sol(R))
      call compute_hadamard_gram_f32(n1, n3, R, U, W, gram)
      do rr = 1, R; gram(rr,rr) = gram(rr,rr) + lambda; end do
      call mttkrp_mode2_f32(n1, n2, n3, R, T, U, W, rhs)
      do i = 1, n2
        sol = rhs(i,:)
        call spotrf('U', R, gram, R, info)
        if (info == 0) then
          call spotrs('U', R, 1, gram, R, sol, R, info)
        else
          allocate(ipiv(R))
          call sgesv(R, 1, gram, R, ipiv, sol, R, info)
          deallocate(ipiv)
        end if
        do rr = 1, R
          out((rr-1)*n2 + i) = sol(rr)
        end do
      end do
      deallocate(rhs, gram, sol)
    else
      allocate(rhs(n3,R), gram(R,R), sol(R))
      call compute_hadamard_gram_f32(n1, n2, R, U, V, gram)
      do rr = 1, R; gram(rr,rr) = gram(rr,rr) + lambda; end do
      call mttkrp_mode3_f32(n1, n2, n3, R, T, U, V, rhs)
      do i = 1, n3
        sol = rhs(i,:)
        call spotrf('U', R, gram, R, info)
        if (info == 0) then
          call spotrs('U', R, 1, gram, R, sol, R, info)
        else
          allocate(ipiv(R))
          call sgesv(R, 1, gram, R, ipiv, sol, R, info)
          deallocate(ipiv)
        end if
        do rr = 1, R
          out((rr-1)*n3 + i) = sol(rr)
        end do
      end do
      deallocate(rhs, gram, sol)
    end if
#else
    if (which == 0) out(1:n1*R) = U(1:n1*R)
    if (which == 1) out(1:n2*R) = V(1:n2*R)
    if (which == 2) out(1:n3*R) = W(1:n3*R)
#endif
  end subroutine lnn_cp_als_f32_full

  subroutine lnn_cp_als_f32_driver(n1, n2, n3, R, T, U, V, W, lambda, iters) &
      bind(C, name='lnn_cp_als_f32_driver')
    integer(c_int), value :: n1, n2, n3, R, iters
    real(c_float), intent(in) :: T(*)
    real(c_float), intent(inout) :: U(*), V(*), W(*)
    real(c_float), value :: lambda
    integer :: it, i
    real(c_float), allocatable :: tmp(:)
    do it = 1, iters
      allocate(tmp(n1*R)); call lnn_cp_als_f32_full(n1,n2,n3,R,T,U,V,W,lambda,0_c_int,tmp)
      do i=1,n1*R; U(i)=tmp(i); end do; deallocate(tmp)
      allocate(tmp(n2*R)); call lnn_cp_als_f32_full(n1,n2,n3,R,T,U,V,W,lambda,1_c_int,tmp)
      do i=1,n2*R; V(i)=tmp(i); end do; deallocate(tmp)
      allocate(tmp(n3*R)); call lnn_cp_als_f32_full(n1,n2,n3,R,T,U,V,W,lambda,2_c_int,tmp)
      do i=1,n3*R; W(i)=tmp(i); end do; deallocate(tmp)
    end do
  end subroutine lnn_cp_als_f32_driver

  subroutine lnn_cp_als_f32_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda_out, tol, max_iters, fit_out, iters_out) &
      bind(C, name='lnn_cp_als_f32_driver_with_lambda')
    integer(c_int), value :: n1, n2, n3, R
    real(c_float), intent(in) :: T(*)
    real(c_float), intent(inout) :: U(*), V(*), W(*)
    real(c_float), intent(out) :: lambda_out(*)
    real(c_float), value :: tol
    integer(c_int), value :: max_iters
    real(c_float), intent(out) :: fit_out(*)
    integer(c_int), intent(out) :: iters_out
    integer :: it, i
    real(c_double) :: prev_err2, err2, Tn2
    real(c_float), allocatable :: U_new(:), V_new(:), W_new(:)
    real(c_float), allocatable :: nu(:), nv(:), nw(:)

    do i = 1, R
      lambda_out(i) = 1.0
    end do
    ! Compute T Frobenius norm squared in double precision
    Tn2 = 0.0d0
    do i = 1, n1*n2*n3
      Tn2 = Tn2 + dble(T(i))*dble(T(i))
    end do
    prev_err2 = cp_recon_error2_lambda_f32(n1, n2, n3, R, T, U, V, W, lambda_out)

    do it = 1, max_iters
      allocate(U_new(n1*R)); allocate(nu(R))
      call lnn_cp_als_f32_full(n1, n2, n3, R, T, U, V, W, 0.0, 0_c_int, U_new)
      call compute_col_norms_f32(n1, R, U_new, nu)
      do i = 1, n1*R; U(i) = U_new(i); end do
      call normalize_columns_flat_f32(n1, R, U)
      do i = 1, R; lambda_out(i) = lambda_out(i) * nu(i); end do
      deallocate(U_new, nu)

      allocate(V_new(n2*R)); allocate(nv(R))
      call lnn_cp_als_f32_full(n1, n2, n3, R, T, U, V, W, 0.0, 1_c_int, V_new)
      call compute_col_norms_f32(n2, R, V_new, nv)
      do i = 1, n2*R; V(i) = V_new(i); end do
      call normalize_columns_flat_f32(n2, R, V)
      do i = 1, R; lambda_out(i) = lambda_out(i) * nv(i); end do
      deallocate(V_new, nv)

      allocate(W_new(n3*R)); allocate(nw(R))
      call lnn_cp_als_f32_full(n1, n2, n3, R, T, U, V, W, 0.0, 2_c_int, W_new)
      call compute_col_norms_f32(n3, R, W_new, nw)
      do i = 1, n3*R; W(i) = W_new(i); end do
      call normalize_columns_flat_f32(n3, R, W)
      do i = 1, R; lambda_out(i) = lambda_out(i) * nw(i); end do
      deallocate(W_new, nw)

      err2 = cp_recon_error2_lambda_f32(n1, n2, n3, R, T, U, V, W, lambda_out)
      if (Tn2 > 0.0d0) then
        fit_out(it) = real(1.0d0 - err2/Tn2, c_float)
      else
        fit_out(it) = 1.0
      end if
      if (Tn2 > 0.0d0) then
        if (real((prev_err2 - err2)/Tn2, c_float) < tol) then
          iters_out = it
          exit
        end if
      else
        if (real(abs(prev_err2 - err2), c_float) < tol) then
          iters_out = it
          exit
        end if
      end if
      prev_err2 = err2
      iters_out = it
    end do
  end subroutine lnn_cp_als_f32_driver_with_lambda
  
  ! Compute Hadamard product of Gram matrices: (A^T A) ⊙ (B^T B)
  subroutine compute_hadamard_gram(nA, nB, R, A, B, gram)
    integer, intent(in) :: nA, nB, R
    real(c_double), intent(in) :: A(*), B(*)
    real(c_double), intent(out) :: gram(R, R)
    real(c_double) :: gramA(R, R), gramB(R, R)
    integer :: rr, s, i
    
    ! Compute A^T A
    do rr = 1, R
      do s = 1, R
        gramA(rr, s) = 0.0d0
        do i = 1, nA
          gramA(rr, s) = gramA(rr, s) + A((rr-1)*nA + i) * A((s-1)*nA + i)
        end do
      end do
    end do
    
    ! Compute B^T B
    do rr = 1, R
      do s = 1, R
        gramB(rr, s) = 0.0d0
        do i = 1, nB
          gramB(rr, s) = gramB(rr, s) + B((rr-1)*nB + i) * B((s-1)*nB + i)
        end do
      end do
    end do
    
    ! Hadamard product
    do rr = 1, R
      do s = 1, R
        gram(rr, s) = gramA(rr, s) * gramB(rr, s)
      end do
    end do
  end subroutine compute_hadamard_gram
  
  ! Khatri-Rao product: C = A ⊙ B
  subroutine khatri_rao_product(nA, nB, R, A, B, C)
    integer, intent(in) :: nA, nB, R
    real(c_double), intent(in) :: A(*), B(*)
    real(c_double), intent(out) :: C(nA*nB, R)
    integer :: i, j, rr, idx
    
    do rr = 1, R
      idx = 1
      do j = 1, nB
        do i = 1, nA
          C(idx, rr) = A((rr-1)*nA + i) * B((rr-1)*nB + j)
          idx = idx + 1
        end do
      end do
    end do
  end subroutine khatri_rao_product
  
  ! Mode-1 MTTKRP: out(i, r) = sum_{j,k} T(i,j,k) * V(j,r) * W(k,r)
  subroutine mttkrp_mode1(n1, n2, n3, R, T, V, W, out)
    integer, intent(in) :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), V(*), W(*)
    real(c_double), intent(out) :: out(n1, R)
    integer :: i, j, k, rr
    integer :: idx
    out = 0.0d0
#ifdef LNN_PROFILE
    real(8) :: t0
    call prof_mark_start('mttkrp1', t0)
#endif
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do i = 1, n1
        do k = 1, n3
          do j = 1, n2
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(i, rr) = out(i, rr) + T(idx) * V((rr-1)*n2 + j) * W((rr-1)*n3 + k)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
#ifdef LNN_PROFILE
    call prof_mark_stop('mttkrp1', t0)
#endif
  end subroutine mttkrp_mode1

  ! Mode-2 MTTKRP: out(j, r) = sum_{i,k} T(i,j,k) * U(i,r) * W(k,r)
  subroutine mttkrp_mode2(n1, n2, n3, R, T, U, W, out)
    integer, intent(in) :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), W(*)
    real(c_double), intent(out) :: out(n2, R)
    integer :: i, j, k, rr
    integer :: idx
    out = 0.0d0
#ifdef LNN_PROFILE
    real(8) :: t02
    call prof_mark_start('mttkrp2', t02)
#endif
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do j = 1, n2
        do k = 1, n3
          do i = 1, n1
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(j, rr) = out(j, rr) + T(idx) * U((rr-1)*n1 + i) * W((rr-1)*n3 + k)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
#ifdef LNN_PROFILE
    call prof_mark_stop('mttkrp2', t02)
#endif
  end subroutine mttkrp_mode2

  ! Mode-3 MTTKRP: out(k, r) = sum_{i,j} T(i,j,k) * U(i,r) * V(j,r)
  subroutine mttkrp_mode3(n1, n2, n3, R, T, U, V, out)
    integer, intent(in) :: n1, n2, n3, R
    real(c_double), intent(in) :: T(*), U(*), V(*)
    real(c_double), intent(out) :: out(n3, R)
    integer :: i, j, k, rr
    integer :: idx
    out = 0.0d0
#ifdef LNN_PROFILE
    real(8) :: t03
    call prof_mark_start('mttkrp3', t03)
#endif
    !$OMP PARALLEL DO collapse(2) private(i,j,k,rr,idx) schedule(static)
    do rr = 1, R
      do k = 1, n3
        do j = 1, n2
          do i = 1, n1
            idx = i + (j-1)*n1 + (k-1)*n1*n2
            out(k, rr) = out(k, rr) + T(idx) * U((rr-1)*n1 + i) * V((rr-1)*n2 + j)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
#ifdef LNN_PROFILE
    call prof_mark_stop('mttkrp3', t03)
#endif
  end subroutine mttkrp_mode3

  ! Unfold tensor along mode-1
  subroutine unfold_tensor_mode1(n1, n2, n3, T, unfolded)
    integer, intent(in) :: n1, n2, n3
    real(c_double), intent(in) :: T(*)
    real(c_double), intent(out) :: unfolded(n1, n2*n3)
    integer :: i, j, k, idx
    
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          idx = i + (j-1)*n1 + (k-1)*n1*n2
          unfolded(i, (k-1)*n2 + j) = T(idx)
        end do
      end do
    end do
  end subroutine unfold_tensor_mode1
  
  ! ============================================================================
  ! TUCKER DECOMPOSITION
  ! ============================================================================
  
  ! Tucker decomposition: T ≈ G ×₁ U ×₂ V ×₃ W
  subroutine lnn_tucker_als_step(n1, n2, n3, r1, r2, r3, T, G, U, V, W, which, out) &
      bind(C, name='lnn_tucker_als_step')
    integer(c_int), value :: n1, n2, n3, r1, r2, r3, which
    real(c_double), intent(in) :: T(*), G(*), U(*), V(*), W(*)
    real(c_double), intent(out) :: out(*)
    
    ! Tucker ALS: update one factor at a time
    ! which: 0=U, 1=V, 2=W, 3=core G
    
    integer :: i
    
    ! Simplified placeholder
    if (which == 0) then
      do i = 1, n1*r1
        out(i) = U(i)
      end do
    else if (which == 1) then
      do i = 1, n2*r2
        out(i) = V(i)
      end do
    else if (which == 2) then
      do i = 1, n3*r3
        out(i) = W(i)
      end do
    else
      do i = 1, r1*r2*r3
        out(i) = G(i)
      end do
    end if
  end subroutine lnn_tucker_als_step
  
  ! ============================================================================
  ! MIXED PRECISION ALS
  ! ============================================================================
  
  ! Mixed precision CP-ALS: f32 inputs, f64 accumulators
  subroutine lnn_cp_als_mixed(n1, n2, n3, R, T32, U32, V32, W32, lambda, which, out32) &
      bind(C, name='lnn_cp_als_mixed')
    integer(c_int), value :: n1, n2, n3, R, which
    real(c_float), intent(in) :: T32(*), U32(*), V32(*), W32(*)
    real(c_double), value :: lambda
    real(c_float), intent(out) :: out32(*)
    
    ! Convert to f64, compute, convert back
    real(c_double), allocatable :: T64(:), U64(:), V64(:), W64(:), out64(:)
    integer :: i, total_size
    
    ! Allocate f64 arrays
    allocate(T64(n1*n2*n3))
    allocate(U64(n1*R))
    allocate(V64(n2*R))
    allocate(W64(n3*R))
    
    if (which == 0) then
      total_size = n1*R
    else if (which == 1) then
      total_size = n2*R
    else
      total_size = n3*R
    end if
    allocate(out64(total_size))
    
    ! Convert to f64
    do i = 1, n1*n2*n3
      T64(i) = real(T32(i), c_double)
    end do
    do i = 1, n1*R
      U64(i) = real(U32(i), c_double)
    end do
    do i = 1, n2*R
      V64(i) = real(V32(i), c_double)
    end do
    do i = 1, n3*R
      W64(i) = real(W32(i), c_double)
    end do
    
    ! Perform ALS in f64
    call lnn_cp_als_full(n1, n2, n3, R, T64, U64, V64, W64, lambda, which, out64)
    
    ! Convert back to f32
    do i = 1, total_size
      out32(i) = real(out64(i), c_float)
    end do
    
    deallocate(T64, U64, V64, W64, out64)
  end subroutine lnn_cp_als_mixed
  
  ! ============================================================================
  ! RIEMANNIAN SGD ON SPD MANIFOLD
  ! ============================================================================
  
  ! Riemannian gradient descent step on SPD manifold
  subroutine lnn_riemannian_sgd_step(n, X, euclidean_grad, learning_rate, out) &
      bind(C, name='lnn_riemannian_sgd_step')
    integer(c_int), value :: n
    real(c_double), intent(in) :: X(*), euclidean_grad(*)
    real(c_double), value :: learning_rate
    real(c_double), intent(out) :: out(*)
    
    ! Riemannian SGD: project Euclidean gradient, then retract
    real(c_double) :: work_x(n, n), work_grad(n, n), riem_grad(n, n)
    real(c_double) :: tangent(n, n), updated(n, n)
    integer :: i, j
    
    ! Convert to matrices
    do j = 1, n
      do i = 1, n
        work_x(i, j) = X((j-1)*n + i)
        work_grad(i, j) = euclidean_grad((j-1)*n + i)
      end do
    end do
    
    ! Project to tangent space: grad_R = X * grad_E * X
    ! (Simplified - full version solves X * grad_R = grad_E)
    call matrix_triple_product(n, work_x, work_grad, work_x, riem_grad)
    
    ! Tangent vector: -learning_rate * grad_R
    do j = 1, n
      do i = 1, n
        tangent(i, j) = -learning_rate * riem_grad(i, j)
      end do
    end do
    
    ! Retract via exponential map (simplified: X + tangent)
    do j = 1, n
      do i = 1, n
        updated(i, j) = work_x(i, j) + tangent(i, j)
      end do
    end do
    
    ! Symmetrize to maintain SPD
    do j = 1, n
      do i = 1, n
        out((j-1)*n + i) = 0.5d0 * (updated(i, j) + updated(j, i))
      end do
    end do
  end subroutine lnn_riemannian_sgd_step
  
  subroutine matrix_triple_product(n, A, B, C, out)
    integer, intent(in) :: n
    real(c_double), intent(in) :: A(n, n), B(n, n), C(n, n)
    real(c_double), intent(out) :: out(n, n)
    real(c_double) :: temp(n, n)
    integer :: i, j, k
    
    ! temp = A * B
    do i = 1, n
      do j = 1, n
        temp(i, j) = 0.0d0
        do k = 1, n
          temp(i, j) = temp(i, j) + A(i, k) * B(k, j)
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
  end subroutine matrix_triple_product

end module advanced_als
