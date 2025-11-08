program spd_ops
  use iso_c_binding
  use tensor_kernels
  implicit none
  integer(c_int), parameter :: n=3
  real(c_double) :: A(n*n), B(n*n), V(n*n), out(n*n), dist
  integer :: i
  ! Simple SPD init: A=I, B=diag(2, 0.5, 1.5), V=I
  A = 0.0d0; B = 0.0d0; V = 0.0d0
  do i = 1, n
    A((i-1)*n + i) = 1.0d0
    V((i-1)*n + i) = 1.0d0
  end do
  B(1) = 2.0d0; B(n+2) = 0.5d0; B(2*n+3) = 1.5d0

  call lnn_spd_distance(n, A, B, dist)
  call lnn_spd_exp_log(n, A, V, out, 0_c_int)
  call lnn_spd_exp_log(n, A, out, V, 1_c_int)
end program spd_ops

