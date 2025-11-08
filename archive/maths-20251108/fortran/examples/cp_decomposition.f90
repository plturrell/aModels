program cp_decomposition
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
end program cp_decomposition

