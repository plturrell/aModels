program topk_search
  use iso_c_binding
  use lnn_kernels
  implicit none
  integer(c_int), parameter :: n=128, m=2000, k=10
  real(c_double), allocatable :: A(:), norms(:), q(:), scores(:)
  integer(c_int), allocatable :: idx(:)
  real(c_double) :: qn
  integer :: i

  allocate(A(n*m), norms(m), q(n), scores(k), idx(k))
  call random_number(A); call random_number(q)
  call lnn_compute_norms(n, m, A, norms)
  call lnn_vector_norm(n, q, qn)
  call lnn_cosine_topk_optimized(n, m, A, norms, q, qn, k, idx, scores)
end program topk_search

