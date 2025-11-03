program test_edge_cases
  use iso_c_binding
  use lnn_kernels
  use tensor_kernels
  implicit none
  integer(c_int) :: n, m, topk
  real(c_double), allocatable :: A(:), q(:), norms(:), scores(:)
  integer(c_int), allocatable :: idx(:)
  real(c_double) :: qn, res
  integer :: i

  ! Zero vector query -> zero scores
  n = 8_c_int; m = 4_c_int; topk = 3_c_int
  allocate(A(n*m), q(n), norms(m), scores(topk), idx(topk))
  A = 0.0d0; do i=1, n*m; A(i) = mod(i,3); end do
  q = 0.0d0
  call lnn_compute_norms(n, m, A, norms)
  call lnn_vector_norm(n, q, qn)
  call lnn_cosine_topk_optimized(n, m, A, norms, q, qn, topk, idx, scores)
  do i = 1, topk
    if (scores(i) /= 0.0d0) then
      print *, 'FAIL: zero query nonzero score'
      stop 1
    end if
  end do

  ! Softmax stability on large values
  block
    integer(c_int) :: mm, nn
    real(c_double), allocatable :: X(:), out(:)
    integer :: j
    mm = 2_c_int; nn = 4_c_int
    allocate(X(mm*nn), out(mm*nn))
    X = 1000.0d0
    call lnn_softmax_batched(mm, nn, X, out)
    do j = 1, mm
      call lnn_vector_norm(nn, out((j-1)*nn+1), res)
      if (.not.(abs(sum(out((j-1)*nn+1:(j-1)*nn+nn)) - 1.0d0) < 1.0d-9)) then
        print *, 'FAIL: softmax row not summing to 1'
        stop 1
      end if
    end do
  end block

  ! SPD distance with identity is zero
  block
    integer(c_int) :: k
    real(c_double) :: I(9), dist
    do k = 1, 9; I(k)=0.0d0; end do
    I(1)=1.0d0; I(5)=1.0d0; I(9)=1.0d0
    call lnn_spd_distance(3_c_int, I, I, dist)
    if (abs(dist) > 1.0d-12) then
      print *, 'FAIL: SPD distance I,I nonzero'
      stop 1
    end if
  end block

  print *, 'Edge cases: OK'
end program test_edge_cases

