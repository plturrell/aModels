#!/usr/bin/env bash
set -euo pipefail

cmd=${1:-auto}
shift || true

detect() {
  local unameOut; unameOut=$(uname -s)
  if [[ "$unameOut" == "Darwin" ]]; then
    echo "darwin"
  else
    echo "linux"
  fi
}

case "$cmd" in
  auto|build)
    if [[ "$(detect)" == "darwin" ]]; then
      echo "[lnn] macOS detected; using Accelerate"
      make smart FC="${FC:-gfortran}"
    else
      echo "[lnn] Linux detected; auto-detecting LAPACK/BLAS"
      make smart FC="${FC:-gfortran}"
    fi
    ;;
  test)
    # Try LAPACK first, fallback to default
    if ldconfig -p 2>/dev/null | grep -qi lapack || [[ -e /usr/lib/x86_64-linux-gnu/liblapack.so ]] || [[ -e /usr/lib64/liblapack.so ]]; then
      make test_lapack FC="${FC:-gfortran}" LNN_USE_BLAS=1 LNN_USE_LAPACK=1
    else
      make test FC="${FC:-gfortran}"
    fi
    ;;
  bench)
    make bench FC="${FC:-gfortran}"
    echo "Run: ./bench/bench_main"
    ;;
  examples)
    make examples FC="${FC:-gfortran}"
    ;;
  *)
    echo "Usage: $0 [auto|build|test|bench|examples]"
    exit 2
    ;;
esac

