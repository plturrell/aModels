module lnn_kernels
  use iso_c_binding
  use lnn_config, only: DEFAULT_EPS, &
                        LNN_SUCCESS, LNN_ERROR_INVALID_INPUT, &
                        LNN_ERROR_ALLOCATION, LNN_ERROR_BACKEND
#ifdef LNN_PROFILE
  use lnn_profiler, only: prof_mark_start, prof_mark_stop
#endif
  implicit none
contains

  ! Enhanced dot with error reporting
  subroutine lnn_dot_safe(n, a, b, res, ierr) bind(C, name="lnn_dot_safe")
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(in)        :: a(*), b(*)
    real(c_double), intent(out)       :: res
    integer(c_int), intent(out)       :: ierr
    integer :: i
    real(c_double) :: acc
    ierr = LNN_SUCCESS
    if (n <= 0) then
      res = 0.0d0
      ierr = LNN_ERROR_INVALID_INPUT
      return
    end if
    acc = 0.0d0
    !$OMP SIMD reduction(+:acc)
    do i = 1, n
      acc = acc + a(i) * b(i)
    end do
    res = acc
  end subroutine lnn_dot_safe

  ! Fused cosine with precomputed norms and clamping
  subroutine lnn_cosine_fused(n, a, a_norm, b, b_norm, res) bind(C, name="lnn_cosine_fused")
    integer(c_int), value           :: n
    real(c_double), intent(in)      :: a(*), b(*)
    real(c_double), value           :: a_norm, b_norm
    real(c_double), intent(out)     :: res
    real(c_double) :: dot_val
    integer :: i
    dot_val = 0.0d0
    !$OMP SIMD reduction(+:dot_val)
    do i = 1, n
      dot_val = dot_val + a(i) * b(i)
    end do
    if (a_norm <= 0.0d0 .or. b_norm <= 0.0d0) then
      res = 0.0d0
    else
      res = dot_val / (a_norm * b_norm)
      if (res > 1.0d0) res = 1.0d0
      if (res < -1.0d0) res = -1.0d0
    end if
  end subroutine lnn_cosine_fused

  ! Compute L2 norms for a batch of row-major vectors
  subroutine lnn_compute_norms(n, m, A, norms) bind(C, name="lnn_compute_norms")
    integer(c_int), value       :: n, m
    real(c_double), intent(in)  :: A(*)
    real(c_double), intent(out) :: norms(*)
    integer :: i, j, base
    real(c_double) :: sum_sq
    if (n <= 0 .or. m <= 0) return
#ifdef LNN_PROFILE
    real(8) :: t0
    call prof_mark_start('compute_norms', t0)
#endif
    !$OMP PARALLEL DO private(i, j, base, sum_sq)
    do i = 0, m-1
      base = i * n
      sum_sq = 0.0d0
      !$OMP SIMD reduction(+:sum_sq)
      do j = 1, n
        sum_sq = sum_sq + A(base + j) * A(base + j)
      end do
      norms(i+1) = sqrt(max(sum_sq, DEFAULT_EPS))
    end do
    !$OMP END PARALLEL DO
#ifdef LNN_PROFILE
    call prof_mark_stop('compute_norms', t0)
#endif
  end subroutine lnn_compute_norms

  ! Single vector L2 norm (double)
  subroutine lnn_vector_norm(n, x, out) bind(C, name="lnn_vector_norm")
    integer(c_int), value       :: n
    real(c_double), intent(in)  :: x(*)
    real(c_double), intent(out) :: out
    integer :: i
    real(c_double) :: sum_sq
    sum_sq = 0.0d0
    if (n <= 0) then
      out = 0.0d0
      return
    end if
    !$OMP SIMD reduction(+:sum_sq)
    do i = 1, n
      sum_sq = sum_sq + x(i) * x(i)
    end do
    out = sqrt(max(sum_sq, DEFAULT_EPS))
  end subroutine lnn_vector_norm

  ! Cosine top-k using precomputed norms for A and q
  subroutine lnn_cosine_topk_optimized(n, m, A, A_norms, q, q_norm, topk, idx, scores) &
      bind(C, name="lnn_cosine_topk_optimized")
    integer(c_int), value           :: n, m, topk
    real(c_double), intent(in)      :: A(*), A_norms(*), q(*)
    real(c_double), value           :: q_norm
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer :: i, j, k, base, minpos
    real(c_double) :: dot_val, s
    if (n <= 0 .or. m <= 0 .or. topk <= 0 .or. q_norm <= 0.0d0) then
      do k = 1, max(topk, 0)
        idx(k) = 0
        scores(k) = 0.0d0
      end do
      return
    end if
#ifdef LNN_PROFILE
    real(8) :: t0
    call prof_mark_start('topk_opt', t0)
#endif
    do k = 1, topk
      idx(k) = 0
      scores(k) = -1.0d300
    end do
    call heap_build(scores, idx, topk)
    do i = 0, m-1
      if (A_norms(i+1) <= 0.0d0) cycle
      base = i * n
      dot_val = 0.0d0
      !$OMP SIMD reduction(+:dot_val)
      do j = 1, n
        dot_val = dot_val + A(base + j) * q(j)
      end do
      s = dot_val / (A_norms(i+1) * q_norm)
      if (s > scores(1)) then
        scores(1) = s
        idx(1) = i + 1
        minpos = 1
        call heap_sift_down(scores, idx, topk, minpos)
      end if
    end do
    call heap_sort_desc(scores, idx, topk)
#ifdef LNN_PROFILE
    call prof_mark_stop('topk_opt', t0)
#endif
  end subroutine lnn_cosine_topk_optimized

  ! Single-precision variants
  subroutine lnn_dot_f32(n, a, b, res) bind(C, name="lnn_dot_f32")
    integer(c_int), value           :: n
    real(c_float), intent(in)       :: a(*), b(*)
    real(c_float), intent(out)      :: res
    integer :: i
    real(c_float) :: acc
    acc = 0.0
    !$OMP SIMD reduction(+:acc)
    do i = 1, n
      acc = acc + a(i) * b(i)
    end do
    res = acc
  end subroutine lnn_dot_f32

  subroutine lnn_compute_norms_f32(n, m, A, norms) bind(C, name="lnn_compute_norms_f32")
    integer(c_int), value       :: n, m
    real(c_float), intent(in)   :: A(*)
    real(c_float), intent(out)  :: norms(*)
    integer :: i, j, base
    real(c_float) :: sum_sq
    if (n <= 0 .or. m <= 0) return
    !$OMP PARALLEL DO private(i, j, base, sum_sq)
    do i = 0, m-1
      base = i * n
      sum_sq = 0.0
      !$OMP SIMD reduction(+:sum_sq)
      do j = 1, n
        sum_sq = sum_sq + A(base + j) * A(base + j)
      end do
      norms(i+1) = sqrt(max(sum_sq, real(DEFAULT_EPS, c_float)))
    end do
    !$OMP END PARALLEL DO
  end subroutine lnn_compute_norms_f32

  ! Single vector L2 norm (float32)
  subroutine lnn_vector_norm_f32(n, x, out) bind(C, name="lnn_vector_norm_f32")
    integer(c_int), value      :: n
    real(c_float), intent(in)  :: x(*)
    real(c_float), intent(out) :: out
    integer :: i
    real(c_float) :: sum_sq
    sum_sq = 0.0
    if (n <= 0) then
      out = 0.0
      return
    end if
    !$OMP SIMD reduction(+:sum_sq)
    do i = 1, n
      sum_sq = sum_sq + x(i) * x(i)
    end do
    out = sqrt(max(sum_sq, real(DEFAULT_EPS, c_float)))
  end subroutine lnn_vector_norm_f32

  subroutine lnn_cosine_fused_f32(n, a, a_norm, b, b_norm, res) bind(C, name="lnn_cosine_fused_f32")
    integer(c_int), value      :: n
    real(c_float), intent(in)  :: a(*), b(*)
    real(c_float), value       :: a_norm, b_norm
    real(c_float), intent(out) :: res
    integer :: i
    real(c_float) :: dot_val
    dot_val = 0.0
    !$OMP SIMD reduction(+:dot_val)
    do i = 1, n
      dot_val = dot_val + a(i) * b(i)
    end do
    if (a_norm <= 0.0 .or. b_norm <= 0.0) then
      res = 0.0
    else
      res = dot_val / (a_norm * b_norm)
      if (res > 1.0) res = 1.0
      if (res < -1.0) res = -1.0
    end if
  end subroutine lnn_cosine_fused_f32

  subroutine lnn_cosine_topk_optimized_f32(n, m, A, A_norms, q, q_norm, topk, idx, scores) &
      bind(C, name="lnn_cosine_topk_optimized_f32")
    integer(c_int), value         :: n, m, topk
    real(c_float), intent(in)     :: A(*), A_norms(*), q(*)
    real(c_float), value          :: q_norm
    integer(c_int), intent(out)   :: idx(*)
    real(c_float), intent(out)    :: scores(*)
    integer :: i, j, k, base, minpos
    integer :: ktop
    real(c_float) :: dot_val, s
    real(c_double) :: sd
    real(c_double), allocatable :: hs_d(:)
    integer(c_int), allocatable :: hi(:)
    if (n <= 0 .or. m <= 0 .or. topk <= 0 .or. q_norm <= 0.0) then
      do k = 1, max(topk, 0)
        idx(k) = 0
        scores(k) = 0.0
      end do
      return
    end if
    ktop = topk
    allocate(hs_d(ktop))
    allocate(hi(ktop))
    do k = 1, ktop
      hi(k) = 0
      hs_d(k) = -1.0d300
    end do
    call heap_build(hs_d, hi, ktop)
    do i = 0, m-1
      if (A_norms(i+1) <= 0.0) cycle
      base = i * n
      dot_val = 0.0
      !$OMP SIMD reduction(+:dot_val)
      do j = 1, n
        dot_val = dot_val + A(base + j) * q(j)
      end do
      s = dot_val / (A_norms(i+1) * q_norm)
      sd = dble(s)
      if (sd > hs_d(1)) then
        hs_d(1) = sd
        hi(1) = i + 1
        minpos = 1
        call heap_sift_down(hs_d, hi, ktop, minpos)
      end if
    end do
    call heap_sort_desc(hs_d, hi, ktop)
    do k = 1, ktop
      scores(k) = real(hs_d(k), c_float)
      idx(k) = hi(k)
    end do
    deallocate(hs_d)
    deallocate(hi)
  end subroutine lnn_cosine_topk_optimized_f32

  ! INT8-optimized variants with precomputed norms
  subroutine lnn_compute_norms_i8(n, m, A8, norms) bind(C, name="lnn_compute_norms_i8")
    integer(c_int), value         :: n, m
    integer(c_int8_t), intent(in) :: A8(*)
    real(c_double), intent(out)   :: norms(*)
    integer :: i, j, base
    real(c_double) :: sum_sq
    if (n <= 0 .or. m <= 0) return
    !$OMP PARALLEL DO private(i, j, base, sum_sq)
    do i = 0, m-1
      base = i * n
      sum_sq = 0.0d0
      !$OMP SIMD reduction(+:sum_sq)
      do j = 1, n
        sum_sq = sum_sq + dble(A8(base + j)) * dble(A8(base + j))
      end do
      norms(i+1) = sqrt(max(sum_sq, DEFAULT_EPS))
    end do
    !$OMP END PARALLEL DO
  end subroutine lnn_compute_norms_i8

  subroutine lnn_cosine_topk_optimized_i8(n, m, A8, A_norms, q, q_norm, topk, idx, scores) &
      bind(C, name="lnn_cosine_topk_optimized_i8")
    integer(c_int), value           :: n, m, topk
    integer(c_int8_t), intent(in)   :: A8(*)
    real(c_double), intent(in)      :: A_norms(*), q(*)
    real(c_double), value           :: q_norm
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer :: i, j, k, base, minpos
    real(c_double) :: dot_val, s
    if (n <= 0 .or. m <= 0 .or. topk <= 0 .or. q_norm <= 0.0d0) then
      do k = 1, max(topk, 0)
        idx(k) = 0
        scores(k) = 0.0d0
      end do
      return
    end if
    do k = 1, topk
      idx(k) = 0
      scores(k) = -1.0d300
    end do
    call heap_build(scores, idx, topk)
    do i = 0, m-1
      if (A_norms(i+1) <= 0.0d0) cycle
      base = i * n
      dot_val = 0.0d0
      !$OMP SIMD reduction(+:dot_val)
      do j = 1, n
        dot_val = dot_val + dble(A8(base + j)) * q(j)
      end do
      s = dot_val / (A_norms(i+1) * q_norm)
      if (s > scores(1)) then
        scores(1) = s
        idx(1) = i + 1
        minpos = 1
        call heap_sift_down(scores, idx, topk, minpos)
      end if
    end do
    call heap_sort_desc(scores, idx, topk)
  end subroutine lnn_cosine_topk_optimized_i8

  ! Min-heap helpers for Top-K (1-based indexing)
  subroutine heap_sift_down(scores, idx, heap_size, pos)
    use iso_c_binding
    implicit none
    integer, intent(in) :: heap_size
    integer, intent(inout) :: pos
    real(c_double), intent(inout) :: scores(*)
    integer(c_int), intent(inout) :: idx(*)
    integer :: left, right, smallest, tmpi
    real(c_double) :: tmps
    do
      left = 2*pos
      right = left + 1
      smallest = pos
      if (left <= heap_size) then
        if (scores(left) < scores(smallest)) smallest = left
      end if
      if (right <= heap_size) then
        if (scores(right) < scores(smallest)) smallest = right
      end if
      if (smallest == pos) exit
      tmps = scores(pos); scores(pos) = scores(smallest); scores(smallest) = tmps
      tmpi = idx(pos); idx(pos) = idx(smallest); idx(smallest) = tmpi
      pos = smallest
    end do
  end subroutine heap_sift_down

  subroutine heap_build(scores, idx, heap_size)
    use iso_c_binding
    implicit none
    integer, intent(in) :: heap_size
    real(c_double), intent(inout) :: scores(*)
    integer(c_int), intent(inout) :: idx(*)
    integer :: i, pos
    if (heap_size <= 1) return
    do i = heap_size/2, 1, -1
      pos = i
      call heap_sift_down(scores, idx, heap_size, pos)
    end do
  end subroutine heap_build

  subroutine heap_sort_desc(scores, idx, heap_size)
    use iso_c_binding
    implicit none
    integer, intent(in) :: heap_size
    real(c_double), intent(inout) :: scores(*)
    integer(c_int), intent(inout) :: idx(*)
    integer :: hs, pos, l, r, tmpi
    real(c_double) :: tmps
    if (heap_size <= 1) return
    hs = heap_size
    do while (hs > 1)
      tmps = scores(1); scores(1) = scores(hs); scores(hs) = tmps
      tmpi = idx(1); idx(1) = idx(hs); idx(hs) = tmpi
      hs = hs - 1
      pos = 1
      call heap_sift_down(scores, idx, hs, pos)
    end do
  end subroutine heap_sort_desc

  ! lnn_dot computes dot product res = sum_i a(i)*b(i)
  subroutine lnn_dot(n, a, b, res) bind(C, name="lnn_dot")
    integer(c_int), value           :: n
    real(c_double), intent(in)      :: a(*), b(*)
    real(c_double), intent(out)     :: res
    integer                         :: i
    real(c_double)                  :: acc
    acc = 0.0d0
    do i = 1, n
      acc = acc + a(i) * b(i)
    end do
    res = acc
  end subroutine lnn_dot

  ! lnn_cosine computes cosine similarity: dot(a,b)/(||a||*||b||)
  subroutine lnn_cosine(n, a, b, res) bind(C, name="lnn_cosine")
    integer(c_int), value           :: n
    real(c_double), intent(in)      :: a(*), b(*)
    real(c_double), intent(out)     :: res
    integer                         :: i
    real(c_double)                  :: da, db, d
    da = 0.0d0
    db = 0.0d0
    d  = 0.0d0
    do i = 1, n
      d  = d  + a(i) * b(i)
      da = da + a(i) * a(i)
      db = db + b(i) * b(i)
    end do
    if (da <= 0.0d0 .or. db <= 0.0d0) then
      res = 0.0d0
    else
      res = d / (sqrt(da) * sqrt(db))
    end if
  end subroutine lnn_cosine

  ! lnn_dot_batch: out(i) = dot(A_i, B_i), i=0..m-1; each row length n
  subroutine lnn_dot_batch(n, m, A, B, out) bind(C, name="lnn_dot_batch")
    integer(c_int), value           :: n, m
    real(c_double), intent(in)      :: A(*), B(*)
    real(c_double), intent(out)     :: out(*)
    integer                         :: i, j
    integer                         :: base
    real(c_double)                  :: acc
    if (n <= 0 .or. m <= 0) return
    do i = 0, m-1
      acc = 0.0d0
      base = i * n
      do j = 0, n-1
        acc = acc + A(base + j + 1) * B(base + j + 1)
      end do
      out(i+1) = acc
    end do
  end subroutine lnn_dot_batch

  ! lnn_cosine_batch: out(i) = dot(A_i,B_i)/(||A_i||*||B_i||)
  subroutine lnn_cosine_batch(n, m, A, B, out) bind(C, name="lnn_cosine_batch")
    integer(c_int), value           :: n, m
    real(c_double), intent(in)      :: A(*), B(*)
    real(c_double), intent(out)     :: out(*)
    integer                         :: i, j, base
    real(c_double)                  :: d, da, db
    if (n <= 0 .or. m <= 0) return
    do i = 0, m-1
      base = i * n
      d = 0.0d0
      da = 0.0d0
      db = 0.0d0
      do j = 0, n-1
        d  = d  + A(base + j + 1) * B(base + j + 1)
        da = da + A(base + j + 1) * A(base + j + 1)
        db = db + B(base + j + 1) * B(base + j + 1)
      end do
      if (da <= 0.0d0 .or. db <= 0.0d0) then
        out(i+1) = 0.0d0
      else
        out(i+1) = d / (sqrt(da) * sqrt(db))
      end if
    end do
  end subroutine lnn_cosine_batch

  ! lnn_matmul performs C = A(m x k) * B(k x n), row-major buffers
  subroutine lnn_matmul(m, n, k, A, B, C) bind(C, name="lnn_matmul")
    integer(c_int), value           :: m, n, k
    real(c_double), intent(in)      :: A(*), B(*)
    real(c_double), intent(out)     :: C(*)
    integer                         :: i, j, p
    real(c_double)                  :: sum
    integer                         :: arow, bcol
#if defined(USE_BLAS) || defined(LNN_USE_BLAS)
    ! Use BLAS dgemm treating row-major buffers as transposed column-major.
    interface
      subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character*1 :: transa, transb
        integer :: m, n, k, lda, ldb, ldc
        real*8 :: alpha, beta
        real*8 :: a(*), b(*), c(*)
      end subroutine dgemm
    end interface
    if (m <= 0 .or. n <= 0 .or. k <= 0) return
    call dgemm('N','N', n, m, k, 1.0d0, B, n, A, k, 0.0d0, C, n)
#else
    if (m <= 0 .or. n <= 0 .or. k <= 0) return
    do i = 0, m-1
      do p = 0, n-1
        sum = 0.0d0
        arow = i * k
        do j = 0, k-1
          ! A index: arow + j, B index: j*n + p, C index: i*n + p
          sum = sum + A(arow + j + 1) * B(j*n + p + 1)
        end do
        C(i*n + p + 1) = sum
      end do
    end do
#endif
  end subroutine lnn_matmul

  ! lnn_project performs Y = A(m x n) * P(n x r), row-major buffers
  subroutine lnn_project(m, n, r, A, P, Y) bind(C, name="lnn_project")
    integer(c_int), value           :: m, n, r
    real(c_double), intent(in)      :: A(*), P(*)
    real(c_double), intent(out)     :: Y(*)
    integer                         :: i, j, k
    real(c_double)                  :: sum
#if defined(USE_BLAS) || defined(LNN_USE_BLAS)
    interface
      subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        character*1 :: transa, transb
        integer :: m, n, k, lda, ldb, ldc
        real*8 :: alpha, beta
        real*8 :: a(*), b(*), c(*)
      end subroutine dgemm
    end interface
    if (m <= 0 .or. n <= 0 .or. r <= 0) return
    call dgemm('N','N', r, m, n, 1.0d0, P, r, A, n, 0.0d0, Y, r)
#else
    if (m <= 0 .or. n <= 0 .or. r <= 0) return
    do i = 0, m-1
      do k = 0, r-1
        sum = 0.0d0
        do j = 0, n-1
          sum = sum + A(i*n + j + 1) * P(j*r + k + 1)
        end do
        Y(i*r + k + 1) = sum
      end do
    end do
#endif
  end subroutine lnn_project

  ! lnn_cosine_topk: given A(m x n) row-major and a single query q(n)
  subroutine lnn_cosine_topk(n, m, A, q, topk, idx, scores) bind(C, name="lnn_cosine_topk")
    integer(c_int), value           :: n, m, topk
    real(c_double), intent(in)      :: A(*), q(*)
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer                         :: i, j, base, k, minpos
    real(c_double)                  :: d, da, dq, s
    if (n <= 0 .or. m <= 0 .or. topk <= 0) return
    dq = 0.0d0
    do j = 1, n
      dq = dq + q(j) * q(j)
    end do
    if (dq <= 0.0d0) then
      do k = 1, topk
        idx(k) = 0
        scores(k) = 0.0d0
      end do
      return
    end if
    dq = sqrt(dq)
    do k = 1, topk
      idx(k) = 0
      scores(k) = -1.0d300
    end do
    call heap_build(scores, idx, topk)
    do i = 0, m-1
      base = i * n
      d = 0.0d0
      da = 0.0d0
      do j = 1, n
        d  = d  + A(base + j) * q(j)
        da = da + A(base + j) * A(base + j)
      end do
      if (da <= 0.0d0) then
        s = -1.0d300
      else
        s = d / (sqrt(da) * dq)
      end if
      if (s > scores(1)) then
        scores(1) = s
        idx(1) = i + 1
        minpos = 1
        call heap_sift_down(scores, idx, topk, minpos)
      end if
    end do
    call heap_sort_desc(scores, idx, topk)
  end subroutine lnn_cosine_topk

  ! lnn_cosine_topk_i8: INT8 docs vs DOUBLE queries
  subroutine lnn_cosine_topk_i8(n, m, A8, q, topk, idx, scores) bind(C, name="lnn_cosine_topk_i8")
    integer(c_int), value           :: n, m, topk
    integer(c_int8_t), intent(in)   :: A8(*)
    real(c_double), intent(in)      :: q(*)
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer                         :: i, j, base, k, pos
    real(c_double)                  :: dq, d, da, s
    dq = 0.0d0
    do j = 1, n
      dq = dq + q(j) * q(j)
    end do
    if (dq <= 0.0d0) then
      do k = 1, topk
        idx(k) = 0; scores(k) = 0.0d0
      end do
      return
    end if
    dq = sqrt(dq)
    do k = 1, topk
      idx(k) = 0
      scores(k) = -1.0d300
    end do
    call heap_build(scores, idx, topk)
    do i = 0, m-1
      base = i * n
      d = 0.0d0
      da = 0.0d0
      do j = 1, n
        d  = d  + dble(A8(base + j)) * q(j)
        da = da + dble(A8(base + j)) * dble(A8(base + j))
      end do
      if (da <= 0.0d0) then
        s = 0.0d0
      else
        s = d / (sqrt(da) * dq)
      end if
      if (s > scores(1)) then
        scores(1) = s
        idx(1) = i + 1
        pos = 1
        call heap_sift_down(scores, idx, topk, pos)
      end if
    end do
    call heap_sort_desc(scores, idx, topk)
  end subroutine lnn_cosine_topk_i8

  ! lnn_cosine_multi_topk_i8: INT8 docs vs DOUBLE queries
  subroutine lnn_cosine_multi_topk_i8(n, m, mq, A8, Qm, topk, idx, scores) bind(C, name="lnn_cosine_multi_topk_i8")
    integer(c_int), value           :: n, m, mq, topk
    integer(c_int8_t), intent(in)   :: A8(*)
    real(c_double), intent(in)      :: Qm(*)
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer                         :: i, j, k, base, pos
    real(c_double), allocatable     :: da(:), dq(:)
    real(c_double), allocatable     :: hs(:)
    integer(c_int), allocatable     :: hi(:)
    real(c_double)                  :: d, s
    integer                         :: off
    if (n <= 0 .or. m <= 0 .or. mq <= 0 .or. topk <= 0) return
    allocate(da(m)); allocate(dq(mq))
    do i = 1, m
      da(i) = 0.0d0
      do k = 1, n
        off = (i-1)*n + k
        da(i) = da(i) + dble(A8(off)) * dble(A8(off))
      end do
      if (da(i) > 0.0d0) da(i) = sqrt(da(i))
    end do
    do j = 1, mq
      dq(j) = 0.0d0
      do k = 1, n
        dq(j) = dq(j) + Qm((j-1)*n + k) * Qm((j-1)*n + k)
      end do
      if (dq(j) > 0.0d0) dq(j) = sqrt(dq(j))
    end do
    allocate(hs(topk)); allocate(hi(topk))
    do j = 1, mq
      base = (j-1) * topk
      do k = 1, topk
        hs(k) = -1.0d300; hi(k) = 0
      end do
      call heap_build(hs, hi, topk)
      do i = 1, m
        if (da(i) <= 0.0d0 .or. dq(j) <= 0.0d0) cycle
        d = 0.0d0
        do k = 1, n
          d = d + dble(A8((i-1)*n + k)) * Qm((j-1)*n + k)
        end do
        s = d / (da(i) * dq(j))
        if (s > hs(1)) then
          hs(1) = s; hi(1) = i; pos = 1
          call heap_sift_down(hs, hi, topk, pos)
        end if
      end do
      call heap_sort_desc(hs, hi, topk)
      do k = 1, topk
        scores(base+k) = hs(k)
        idx(base+k) = hi(k)
      end do
    end do
    deallocate(hs); deallocate(hi)
    deallocate(da); deallocate(dq)
  end subroutine lnn_cosine_multi_topk_i8

  ! lnn_cosine_multi_topk for DOUBLE docs/queries
  subroutine lnn_cosine_multi_topk(n, m, mq, A, Qm, topk, idx, scores) bind(C, name="lnn_cosine_multi_topk")
    integer(c_int), value           :: n, m, mq, topk
    real(c_double), intent(in)      :: A(*), Qm(*)
    integer(c_int), intent(out)     :: idx(*)
    real(c_double), intent(out)     :: scores(*)
    integer                         :: i, j, k, base
    integer                         :: minpos
    real(c_double), allocatable     :: D(:,:), da(:), dq(:)
    real(c_double), allocatable     :: hs(:)
    integer(c_int), allocatable     :: hi(:)
    if (n <= 0 .or. m <= 0 .or. mq <= 0 .or. topk <= 0) return
    allocate(D(m,mq)); allocate(da(m)); allocate(dq(mq))
    do i = 1, m
      da(i) = 0.0d0
      do k = 1, n
        da(i) = da(i) + A((i-1)*n + k) * A((i-1)*n + k)
      end do
      if (da(i) > 0.0d0) da(i) = sqrt(da(i))
    end do
    do j = 1, mq
      dq(j) = 0.0d0
      do k = 1, n
        dq(j) = dq(j) + Qm((j-1)*n + k) * Qm((j-1)*n + k)
      end do
      if (dq(j) > 0.0d0) dq(j) = sqrt(dq(j))
    end do
    do i = 1, m
      do j = 1, mq
        D(i,j) = 0.0d0
        do k = 1, n
          D(i,j) = D(i,j) + A((i-1)*n + k) * Qm((j-1)*n + k)
        end do
      end do
    end do
    allocate(hs(topk)); allocate(hi(topk))
    do j = 1, mq
      base = (j-1) * topk
      do k = 1, topk
        hs(k) = -1.0d300; hi(k) = 0
      end do
      call heap_build(hs, hi, topk)
      do i = 1, m
        if (da(i) <= 0.0d0 .or. dq(j) <= 0.0d0) cycle
        D(i,j) = D(i,j) / (da(i) * dq(j))
        if (D(i,j) > hs(1)) then
          hs(1) = D(i,j); hi(1) = i; minpos = 1
          call heap_sift_down(hs, hi, topk, minpos)
        end if
      end do
      call heap_sort_desc(hs, hi, topk)
      do k = 1, topk
        scores(base+k) = hs(k)
        idx(base+k) = hi(k)
      end do
    end do
    deallocate(hs); deallocate(hi)
    deallocate(D); deallocate(da); deallocate(dq)
  end subroutine lnn_cosine_multi_topk

end module lnn_kernels
