module lnn_config
  use iso_c_binding
  implicit none

  ! Global numeric constants
  real(c_double), parameter :: DEFAULT_EPS = 1.0d-12

  ! Generic error/status codes
  integer, parameter :: LNN_SUCCESS            = 0
  integer, parameter :: LNN_ERROR_INVALID_INPUT = -1
  integer, parameter :: LNN_ERROR_ALLOCATION    = -2
  integer, parameter :: LNN_ERROR_BACKEND       = -3

  ! SPD eigensolver configuration
  integer, parameter :: SPD_EIG_DSYEVD = 1
  integer, parameter :: SPD_EIG_DSYEVR = 2
  integer :: SPD_EIG_PATH = SPD_EIG_DSYEVD
  real(c_double) :: SPD_ABSTOL = 0.0d0

  logical :: LNN_ENV_INIT = .false.

  ! Compile-time feature flags (reflected as parameters)
#if defined(USE_BLAS) || defined(LNN_USE_BLAS)
  logical, parameter :: HAS_BLAS = .true.
#else
  logical, parameter :: HAS_BLAS = .false.
#endif

#if defined(USE_LAPACK) || defined(LNN_USE_LAPACK)
  logical, parameter :: HAS_LAPACK = .true.
#else
  logical, parameter :: HAS_LAPACK = .false.
#endif

contains

  subroutine lnn_read_env()
    character(len=128) :: val
    integer :: stat, len
    if (LNN_ENV_INIT) return
    ! SPD_EIG_PATH: 1=DSYEVD, 2=DSYEVR
    call get_environment_variable('LNN_SPD_EIG_PATH', val, length=len, status=stat)
    if (stat == 0 .and. len > 0) then
      select case (trim(val))
      case ('1')
        SPD_EIG_PATH = SPD_EIG_DSYEVD
      case ('2')
        SPD_EIG_PATH = SPD_EIG_DSYEVR
      end select
    end if
    call get_environment_variable('LNN_SPD_ABSTOL', val, length=len, status=stat)
    if (stat == 0 .and. len > 0) then
      read(val,*,err=10) SPD_ABSTOL
    end if
10  continue
    LNN_ENV_INIT = .true.
  end subroutine lnn_read_env

end module lnn_config
