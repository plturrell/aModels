program bench_main
  use iso_c_binding
  use lnn_kernels
  use advanced_als
  use lnn_profiler
  implicit none

  integer, parameter :: dp = kind(1.0d0)
  integer(c_int) :: n, m, topk
  real(c_double), allocatable :: A(:), q(:), norms(:), scores(:)
  integer(c_int), allocatable :: idx(:)
  real(c_double) :: t0, t1, seconds
  integer :: reps, i
  character(len=256) :: csv

  n = 256_c_int
  m = 5000_c_int
  topk = 20_c_int
  reps = 10

  allocate(A(n*m), q(n), norms(m), scores(topk), idx(topk))

  call random_seed()
  call random_number(A)
  call random_number(q)

  call cpu_time(t0)
  call lnn_compute_norms(n, m, A, norms)
  call cpu_time(t1)
  seconds = t1 - t0
  print *, 'compute_norms: ', seconds, ' s'

  call cpu_time(t0)
  do i = 1, reps
    call lnn_cosine_topk_optimized(n, m, A, norms, q, 1.0d0, topk, idx, scores)
  end do
  call cpu_time(t1)
  seconds = (t1 - t0) / real(reps, dp)
  print *, 'topk_optimized (avg): ', seconds, ' s'

  ! MTTKRP timing (mode-1)
  block
    integer(c_int) :: n1, n2, n3, R
    real(c_double), allocatable :: T(:), V(:), W(:), out(:,:)
    real(c_float), allocatable :: Tf(:), Vf(:), Wf(:), outf(:,:)
    n1 = 128_c_int; n2 = 64_c_int; n3 = 32_c_int; R = 32_c_int
    allocate(T(n1*n2*n3), V(n2*R), W(n3*R), out(n1,R))
    allocate(Tf(n1*n2*n3), Vf(n2*R), Wf(n3*R), outf(n1,R))
    call random_number(T); call random_number(V); call random_number(W)
    ! Convert to f32
    do i = 1, n1*n2*n3; Tf(i) = real(T(i), c_float); end do
    do i = 1, n2*R; Vf(i) = real(V(i), c_float); end do
    do i = 1, n3*R; Wf(i) = real(W(i), c_float); end do
    call cpu_time(t0)
    do i = 1, reps
      call mttkrp_mode1(n1, n2, n3, R, T, V, W, out)
    end do
    call cpu_time(t1)
    seconds = (t1 - t0) / real(reps, dp)
    print *, 'mttkrp_mode1 (avg): ', seconds, ' s'
    call cpu_time(t0)
    do i = 1, reps
      call mttkrp_mode1_f32(n1, n2, n3, R, Tf, Vf, Wf, outf)
    end do
    call cpu_time(t1)
    seconds = (t1 - t0) / real(reps, dp)
    print *, 'mttkrp_mode1_f32 (avg): ', seconds, ' s'
  end block

  ! Solve timing: Cholesky vs GE
  block
    integer :: Rn, info
    real(c_double), allocatable :: G(:,:), b(:), x(:), Gc(:,:)
    integer, allocatable :: ipiv(:)
    Rn = 128
    allocate(G(Rn,Rn), b(Rn), x(Rn), Gc(Rn,Rn))
    call random_number(G)
    G = matmul(transpose(G), G) + 1.0d-3*Rn*reshape([ (1.0d0, i=1,Rn*Rn) ], [Rn,Rn])
    call random_number(b)
    Gc = G
    allocate(ipiv(Rn))
    call cpu_time(t0)
    do i = 1, reps
      x = b
      call dpotrf('U', Rn, Gc, Rn, info)
      call dpotrs('U', Rn, 1, Gc, Rn, x, Rn, info)
      Gc = G
    end do
    call cpu_time(t1)
    print *, 'solve DPOTRF/DPOTRS (avg): ', (t1 - t0)/real(reps,dp), ' s'
    call cpu_time(t0)
    do i = 1, reps
      x = b
      call dgesv(Rn, 1, Gc, Rn, ipiv, x, Rn, info)
      Gc = G
    end do
    call cpu_time(t1)
  print *, 'solve DGESV (avg): ', (t1 - t0)/real(reps,dp), ' s'
  deallocate(ipiv)
  end block

  ! End-to-end ALS (double) timing and final fit
  block
    integer(c_int) :: n1, n2, n3, R, iters
    real(c_double), allocatable :: T(:), U(:), V(:), W(:), lambda(:), fit(:)
    real(c_double) :: tfit0, tfit1
    n1=64_c_int; n2=48_c_int; n3=32_c_int; R=8_c_int; iters=10_c_int
    allocate(T(n1*n2*n3), U(n1*R), V(n2*R), W(n3*R), lambda(R), fit(iters))
    call random_number(T); call random_number(U); call random_number(V); call random_number(W)
    call cpu_time(tfit0)
    call lnn_cp_als_driver_with_lambda(n1,n2,n3,R,T,U,V,W, lambda, 1.0d-6, iters, fit, iters)
    call cpu_time(tfit1)
    print *, 'ALS (double) sweeps: ', iters, ' time: ', (tfit1-tfit0), ' s', ' final fit: ', fit(iters)
  end block

  ! Optional profiler CSV dump via env var LNN_PROFILE_CSV
  if (get_environment_variable('LNN_PROFILE_CSV', csv) .eq. 0) then
    call prof_report_csv(trim(csv))
  end if

  deallocate(A, q, norms, scores, idx)
end program bench_main
