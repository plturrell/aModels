module lnn_profiler
  implicit none
#ifdef LNN_PROFILE
  integer, parameter :: MAX_EVENTS = 64
  character(len=64) :: names(MAX_EVENTS)
  real(8) :: total(MAX_EVENTS)
  integer :: counts(MAX_EVENTS)
  integer :: used = 0
#endif
contains
  subroutine prof_reset()
#ifdef LNN_PROFILE
    integer :: i
    used = 0
    do i = 1, MAX_EVENTS
      names(i) = ''
      total(i) = 0.0d0
      counts(i) = 0
    end do
#endif
  end subroutine prof_reset

  subroutine prof_mark_start(label, t0)
    character(len=*), intent(in) :: label
    real(8), intent(out) :: t0
#ifdef LNN_PROFILE
    call cpu_time(t0)
#else
    t0 = 0.0d0
#endif
  end subroutine prof_mark_start

  subroutine prof_mark_stop(label, t0)
    character(len=*), intent(in) :: label
    real(8), intent(in) :: t0
#ifdef LNN_PROFILE
    real(8) :: t1
    integer :: i
    call cpu_time(t1)
    do i = 1, used
      if (trim(names(i)) == trim(label)) then
        total(i) = total(i) + (t1 - t0)
        counts(i) = counts(i) + 1
        return
      end if
    end do
    if (used < MAX_EVENTS) then
      used = used + 1
      names(used) = label
      total(used) = (t1 - t0)
      counts(used) = 1
    end if
#endif
  end subroutine prof_mark_stop

  subroutine prof_report()
#ifdef LNN_PROFILE
    integer :: i
    do i = 1, used
      write(*,'(A,1X,F10.6,1X,A,1X,I8)') trim(names(i)), total(i), 's', counts(i)
    end do
#endif
  end subroutine prof_report

  subroutine prof_report_csv(filename)
    character(len=*), intent(in) :: filename
#ifdef LNN_PROFILE
    integer :: i, u
    open(newunit=u, file=filename, status='replace', action='write')
    write(u,'(A)') 'event,total_seconds,count'
    do i = 1, used
      write(u,'(A,",",F12.6,",",I0)') trim(names(i)), total(i), counts(i)
    end do
    close(u)
#endif
  end subroutine prof_report_csv
end module lnn_profiler
