program test_main
  use iso_c_binding
  use lnn_kernels
  use advanced_als
  implicit none

  integer(c_int) :: n, m, topk
  real(c_double), allocatable :: A(:), q(:), A_norms(:), scores(:)
  integer(c_int), allocatable :: idx(:)
  real(c_double) :: dot_res, q_norm, fused_res
  integer(c_int) :: ierr
  integer :: i

  ! Basic sizes
  n = 4_c_int
  m = 3_c_int
  topk = 2_c_int

  allocate(A(n*m), q(n), A_norms(m), scores(topk), idx(topk))

  ! Fill A as 3 rows: [1,2,3,4], [4,3,2,1], [1,1,1,1]
  A = [1d0,2d0,3d0,4d0, &
        4d0,3d0,2d0,1d0, &
        1d0,1d0,1d0,1d0]
  ! Query q = [1,0,0,0]
  q = [1d0,0d0,0d0,0d0]

  call lnn_dot_safe(n, A(1), q, dot_res, ierr)
  if (ierr /= 0_c_int) then
    print *, 'FAIL: lnn_dot_safe ierr=', ierr
    stop 1
  end if
  if (abs(dot_res - 1.0d0) > 1.0d-12) then
    print *, 'FAIL: lnn_dot_safe result=', dot_res
    stop 1
  end if

  call lnn_compute_norms(n, m, A, A_norms)

  call lnn_vector_norm(n, q, q_norm)

  call lnn_cosine_fused(n, A(1), A_norms(1), q, q_norm, fused_res)
  if (fused_res <= 0.0d0) then
    print *, 'FAIL: lnn_cosine_fused result=', fused_res
    stop 1
  end if

  call lnn_cosine_topk_optimized(n, m, A, A_norms, q, q_norm, topk, idx, scores)
  if (idx(1) <= 0_c_int .or. scores(1) <= 0.0d0) then
    print *, 'FAIL: lnn_cosine_topk_optimized top-1 invalid'
    stop 1
  end if

  ! Quick INT8 test
  block
    integer(c_int8_t), allocatable :: A8(:)
    real(c_double), allocatable :: A8n(:), s2(:)
    integer(c_int), allocatable :: i2(:)
    allocate(A8(n*m))
    allocate(A8n(m))
    allocate(s2(topk))
    allocate(i2(topk))
    ! A8 is same as A but cast to int8
    do i = 1, n*m
      A8(i) = int(A(i), kind=c_int8_t)
    end do
    call lnn_compute_norms_i8(n, m, A8, A8n)
    call lnn_cosine_topk_optimized_i8(n, m, A8, A8n, q, q_norm, topk, i2, s2)
    if (i2(1) <= 0_c_int .or. s2(1) <= 0.0d0) then
      print *, 'FAIL: lnn_cosine_topk_optimized_i8 top-1 invalid'
      stop 1
    end if
  end block

  ! f32 quick sanity
  block
    real(c_float), allocatable :: Af(:), qf(:), nf(:), sf(:)
    integer(c_int), allocatable :: idf(:)
    real(c_float) :: rf, qnf
    allocate(Af(n*m), qf(n), nf(m), sf(topk), idf(topk))
    do i = 1, n*m
      Af(i) = real(A(i), c_float)
    end do
    do i = 1, n
      qf(i) = real(q(i), c_float)
    end do
    call lnn_dot_f32(n, Af(1), qf, rf)
    call lnn_compute_norms_f32(n, m, Af, nf)
    call lnn_vector_norm_f32(n, qf, qnf)
    call lnn_cosine_topk_optimized_f32(n, m, Af, nf, qf, qnf, topk, idf, sf)
    if (idf(1) <= 0_c_int .or. sf(1) <= 0.0) then
      print *, 'FAIL: lnn_cosine_topk_optimized_f32 top-1 invalid'
      stop 1
    end if
  end block

  ! Randomized larger case (sanity only)
  block
    integer, parameter :: n2 = 128, m2 = 500, k2 = 32
    real(c_double), allocatable :: A2(:), q2(:), norms2(:), scores2(:)
    integer(c_int), allocatable :: idx2(:)
    integer :: j
    real(c_double) :: prev
    allocate(A2(n2*m2), q2(n2), norms2(m2), scores2(10), idx2(10))
    call random_seed()
    call random_number(A2)
    call random_number(q2)
    call lnn_compute_norms(n2, m2, A2, norms2)
    call lnn_vector_norm(n2, q2, q_norm)
    call lnn_cosine_topk_optimized(n2, m2, A2, norms2, q2, q_norm, 10, idx2, scores2)
    prev = scores2(1)
    do j = 2, 10
      if (scores2(j) > prev + 1.0d-12) then
        print *, 'FAIL: scores not nonincreasing at j=', j
        stop 1
      end if
      prev = scores2(j)
    end do
  end block

  ! Matmul sanity (row-major): A(3x4) * B(4x2)
  block
    integer(c_int) :: mm, nn, kk
    real(c_double) :: A3(12), B3(8), C3(6)
    mm = 3_c_int; kk = 4_c_int; nn = 2_c_int
    A3 = [1d0,2d0,3d0,4d0, 5d0,6d0,7d0,8d0, 9d0,10d0,11d0,12d0]
    B3 = [1d0, 0d0,  &
          0d0, 1d0,  &
          0d0, 0d0,  &
          0d0, 0d0]
    call lnn_matmul(mm, nn, kk, A3, B3, C3)
    if (abs(C3(1) - 1d0) > 1d-9 .or. abs(C3(2) - 2d0) > 1d-9) then
      print *, 'FAIL: matmul first row unexpected', C3(1), C3(2)
      stop 1
    end if
    if (abs(C3(3) - 5d0) > 1d-9 .or. abs(C3(4) - 6d0) > 1d-9) then
      print *, 'FAIL: matmul second row unexpected', C3(3), C3(4)
      stop 1
    end if
  end block

  ! CP-ALS step delegates to full implementation
  block
    integer(c_int) :: n1,n2,n3,R,which
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), out_full(:), out_step(:)
    integer :: sz
    n1 = 3_c_int; n2 = 2_c_int; n3 = 2_c_int; R = 2_c_int; which = 0_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R))
    allocate(out_full(n1*R), out_step(n1*R))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_full)
    call lnn_cp_als_step(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_step)
    do i = 1, n1*R
      if (abs(out_full(i) - out_step(i)) > 1.0d-6) then
        print *, 'FAIL: cp_als_step mismatch at i=', i, out_full(i), out_step(i)
        stop 1
      end if
    end do
  end block

  ! Mode-2 step equals full (V update)
  block
    integer(c_int) :: n1,n2,n3,R,which
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), out_full(:), out_step(:)
    integer :: sz
    n1 = 3_c_int; n2 = 3_c_int; n3 = 2_c_int; R = 2_c_int; which = 1_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R))
    allocate(out_full(n2*R), out_step(n2*R))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_full)
    call lnn_cp_als_step(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_step)
    do i = 1, n2*R
      if (abs(out_full(i) - out_step(i)) > 1.0d-6) then
        print *, 'FAIL: cp_als_step (mode-2) mismatch at i=', i, out_full(i), out_step(i)
        stop 1
      end if
    end do
  end block

  ! Mode-3 step equals full (W update)
  block
    integer(c_int) :: n1,n2,n3,R,which
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), out_full(:), out_step(:)
    integer :: sz
    n1 = 3_c_int; n2 = 2_c_int; n3 = 3_c_int; R = 2_c_int; which = 2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R))
    allocate(out_full(n3*R), out_step(n3*R))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call lnn_cp_als_full(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_full)
    call lnn_cp_als_step(n1, n2, n3, R, T, U, V, W, 1.0d-4, which, out_step)
    do i = 1, n3*R
      if (abs(out_full(i) - out_step(i)) > 1.0d-6) then
        print *, 'FAIL: cp_als_step (mode-3) mismatch at i=', i, out_full(i), out_step(i)
        stop 1
      end if
    end do
  end block

  ! SPD distance (only when LAPACK is available)
  ! dist(I, diag(2, 0.5)) = sqrt( (log 2)^2 + (log 0.5)^2 )
  !                        = sqrt(2) * log 2
#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
  block
    integer(c_int) :: n
    real(c_double) :: A(4), B(4), dist, expected
    n = 2_c_int
    A = [1d0, 0d0, 0d0, 1d0]
    B = [2d0, 0d0, 0d0, 0.5d0]
    call lnn_spd_distance(n, A, B, dist)
    expected = sqrt(2.0d0) * log(2.0d0)
    if (abs(dist - expected) > 1.0d-8) then
      print *, 'FAIL: spd_distance', dist, expected
      stop 1
    end if
  end block

  ! CP-ALS driver reduces reconstruction error on synthetic data
  block
    integer(c_int) :: n1, n2, n3, R
    integer :: i, j, k, r
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), U0(:), V0(:), W0(:)
    real(c_double) :: err0, err1, x
    n1 = 5_c_int; n2 = 4_c_int; n3 = 3_c_int; R = 2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), U0(n1*R), V0(n2*R), W0(n3*R))
    call random_number(U); call random_number(V); call random_number(W)
    ! Build exact rank-R tensor T = sum_r U(:,r) ⊗ V(:,r) ⊗ W(:,r)
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do r = 1, R
            x = x + U((r-1)*n1 + i) * V((r-1)*n2 + j) * W((r-1)*n3 + k)
          end do
          T(i + (j-1)*n1 + (k-1)*n1*n2) = x
        end do
      end do
    end do
    ! Random initial factors U0,V0,W0
    call random_number(U0); call random_number(V0); call random_number(W0)
    ! Compute initial reconstruction error
    err0 = 0.0d0
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do r = 1, R
            x = x + U0((r-1)*n1 + i) * V0((r-1)*n2 + j) * W0((r-1)*n3 + k)
          end do
          err0 = err0 + (T(i + (j-1)*n1 + (k-1)*n1*n2) - x)**2
        end do
      end do
    end do
    call lnn_cp_als_driver(n1, n2, n3, R, T, U0, V0, W0, 1.0d-6, 5_c_int)
    ! Compute reconstruction error after a few ALS iterations
    err1 = 0.0d0
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          x = 0.0d0
          do r = 1, R
            x = x + U0((r-1)*n1 + i) * V0((r-1)*n2 + j) * W0((r-1)*n3 + k)
          end do
          err1 = err1 + (T(i + (j-1)*n1 + (k-1)*n1*n2) - x)**2
        end do
      end do
    end do
    if (err1 > err0 - 1.0d-10) then
      print *, 'FAIL: ALS driver did not reduce error', err0, err1
      stop 1
    end if
  end block

  ! CP-ALS driver with lambda returns increasing fit and reasonable lambda
  block
    integer(c_int) :: n1, n2, n3, R, iters
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), lambda(:), fit(:)
    integer :: i
    n1 = 5_c_int; n2 = 4_c_int; n3 = 3_c_int; R = 2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R), fit(10))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call lnn_cp_als_driver_with_lambda(n1, n2, n3, R, T, U, V, W, lambda, 1.0d-8, 10_c_int, fit, iters)
    do i = 2, iters
      if (fit(i) + 1.0d-12 < fit(i-1)) then
        print *, 'FAIL: fit not nondecreasing at i=', i, fit(i-1), fit(i)
        stop 1
      end if
    end do
    do i = 1, R
      if (lambda(i) <= 0.0d0) then
        print *, 'FAIL: lambda nonpositive at i=', i, lambda(i)
        stop 1
      end if
    end do
  end block

  ! Fit and residual via reconstruction utilities
  block
    integer(c_int) :: n1,n2,n3,R
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), lambda(:), That(:)
    real(c_double) :: res, fit, tn2, err2
    integer :: i
    n1 = 3_c_int; n2 = 3_c_int; n3 = 2_c_int; R = 2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R), That(n1*n2*n3))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    do i = 1, R; lambda(i) = 1.0d0; end do
    call lnn_cp_reconstruct(n1, n2, n3, R, U, V, W, lambda, That)
    ! Compute residual norm explicitly
    err2 = 0.0d0
    do i = 1, n1*n2*n3
      err2 = err2 + (T(i) - That(i))**2
    end do
    call lnn_cp_residual_norm(n1, n2, n3, R, T, U, V, W, lambda, res)
    if (abs(res - sqrt(err2)) > 1.0d-9) then
      print *, 'FAIL: cp_residual_norm mismatch'
      stop 1
    end if
    tn2 = 0.0d0
    do i = 1, n1*n2*n3
      tn2 = tn2 + T(i)*T(i)
    end do
    call lnn_cp_fit(n1, n2, n3, R, T, U, V, W, lambda, fit)
    if (tn2 > 0.0d0) then
      if (abs(fit - (1.0d0 - err2/tn2)) > 1.0d-9) then
        print *, 'FAIL: cp_fit mismatch'
        stop 1
      end if
    end if
  end block

  ! Normal-equation residuals decrease after one ALS sweep
  block
    integer(c_int) :: n1,n2,n3,R
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), lambda(:)
    real(c_double) :: ru0, rv0, rw0, ru1, rv1, rw1
    integer :: i
    n1 = 6_c_int; n2 = 5_c_int; n3 = 4_c_int; R = 2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    do i = 1, R; lambda(i) = 1.0d0; end do
    call lnn_cp_normal_eq_residuals(n1, n2, n3, R, T, U, V, W, lambda, ru0, rv0, rw0)
    call lnn_cp_als_driver(n1, n2, n3, R, T, U, V, W, 1.0d-6, 1_c_int)
    call lnn_cp_normal_eq_residuals(n1, n2, n3, R, T, U, V, W, lambda, ru1, rv1, rw1)
    if (ru1 > ru0 + 1.0d-10 .or. rv1 > rv0 + 1.0d-10 .or. rw1 > rw0 + 1.0d-10) then
      print *, 'FAIL: normal-equation residuals did not decrease'
      stop 1
    end if
  end block

  ! f32 ALS driver reduces reconstruction error on small synthetic case
  block
    integer(c_int) :: n1,n2,n3,R
    real(c_float), allocatable :: T(:), U(:), V(:), W(:)
    real(c_float) :: err0, err1, x
    integer :: i,j,k,r
    n1=4_c_int; n2=3_c_int; n3=2_c_int; R=2_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R))
    call random_number(U); call random_number(V); call random_number(W)
    do k=1,n3; do j=1,n2; do i=1,n1
      x=0.0
      do r=1,R
        x = x + U((r-1)*n1+i)*V((r-1)*n2+j)*W((r-1)*n3+k)
      end do
      T(i+(j-1)*n1+(k-1)*n1*n2) = x
    end do; end do; end do
    call random_number(U); call random_number(V); call random_number(W)
    err0 = 0.0
    do k=1,n3; do j=1,n2; do i=1,n1
      x=0.0; do r=1,R
        x = x + U((r-1)*n1+i)*V((r-1)*n2+j)*W((r-1)*n3+k)
      end do
      err0 = err0 + (T(i+(j-1)*n1+(k-1)*n1*n2) - x)**2
    end do; end do; end do
    call lnn_cp_als_f32_driver(n1,n2,n3,R,T,U,V,W,1.0e-6,5_c_int)
    err1 = 0.0
    do k=1,n3; do j=1,n2; do i=1,n1
      x=0.0; do r=1,R
        x = x + U((r-1)*n1+i)*V((r-1)*n2+j)*W((r-1)*n3+k)
      end do
      err1 = err1 + (T(i+(j-1)*n1+(k-1)*n1*n2) - x)**2
    end do; end do; end do
    if (real(err1,kind=8) > real(err0,kind=8) - 1.0d-6) then
      print *, 'FAIL: f32 ALS did not reduce error'
      stop 1
    end if
  end block

  ! f32 ALS driver with lambda/fit; fit should be nondecreasing
  block
    integer(c_int) :: n1,n2,n3,R,iters
    real(c_float), allocatable :: T(:), U(:), V(:), W(:), lambda(:), fit(:)
    integer :: i
    n1=4_c_int; n2=3_c_int; n3=2_c_int; R=2_c_int; iters=8_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R), fit(iters))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call lnn_cp_als_f32_driver_with_lambda(n1,n2,n3,R,T,U,V,W, lambda, 1.0e-6, iters, fit, iters)
    do i = 2, iters
      if (fit(i) + 1.0e-7 < fit(i-1)) then
        print *, 'FAIL: f32 with_lambda fit not nondecreasing', fit(i-1), fit(i)
        stop 1
      end if
    end do
  end block
#endif

  print *, 'All tests passed.'
end program test_main
