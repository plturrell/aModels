#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
outdir="$here/lib"
mkdir -p "$outdir"

# Prefer Homebrew gfortran if present
GFORTRAN_BIN="${GFORTRAN_BIN:-}"
if [[ -z "${GFORTRAN_BIN}" ]]; then
  for c in gfortran-14 gfortran-13 gfortran; do
    if command -v "$c" >/dev/null 2>&1; then GFORTRAN_BIN="$c"; break; fi
  done
fi

if [[ -z "${GFORTRAN_BIN}" ]]; then
  echo "Error: gfortran not found. Install via Homebrew: brew install gcc" >&2
  exit 1
fi

echo "Using Fortran compiler: $GFORTRAN_BIN"

LDFLAGS_EXTRA=""
CPPFLAGS_EXTRA=" -cpp"

if [[ "${LNN_USE_ACCELERATE:-}" == "1" ]]; then
  CPPFLAGS_EXTRA+=" -DLNN_USE_BLAS -DUSE_BLAS"
  LDFLAGS_EXTRA+=" -Wl,-framework,Accelerate"
elif [[ "${LNN_USE_BLAS:-}" == "1" ]]; then
  CPPFLAGS_EXTRA+=" -cpp -DLNN_USE_BLAS -DUSE_BLAS"
  if command -v brew >/dev/null 2>&1; then
    OB_PREFIX=$(brew --prefix openblas 2>/dev/null || true)
    if [[ -n "$OB_PREFIX" && -d "$OB_PREFIX/lib" ]]; then
      LDFLAGS_EXTRA+=" -L$OB_PREFIX/lib -lopenblas"
    fi
  fi
  if [[ -z "$LDFLAGS_EXTRA" ]]; then
    LDFLAGS_EXTRA+=" -lopenblas"
  fi
fi

# Optional LAPACK toggle (when not using Accelerate)
if [[ "${LNN_USE_LAPACK:-}" == "1" ]]; then
  CPPFLAGS_EXTRA+=" -DLNN_USE_LAPACK -DUSE_LAPACK"
  LDFLAGS_EXTRA+=" -llapack"
fi

COMMON_FLAGS="-O3 -ffast-math -fPIC"
if [[ "$(uname -s)" == "Darwin" ]]; then
  COMMON_FLAGS+=" -mcpu=apple-m3"
else
  COMMON_FLAGS+=" -march=native -mtune=native"
fi

case "$(uname -s)" in
  Darwin)
    "$GFORTRAN_BIN" $COMMON_FLAGS -dynamiclib -o "$outdir/liblnn.dylib" $CPPFLAGS_EXTRA "$here/lnn_config.f90" "$here/lnn_kernels.f90" $LDFLAGS_EXTRA
    echo "Built: $outdir/liblnn.dylib"
    ;;
  Linux)
    "$GFORTRAN_BIN" $COMMON_FLAGS -shared -o "$outdir/liblnn.so" $CPPFLAGS_EXTRA "$here/lnn_config.f90" "$here/lnn_kernels.f90" $LDFLAGS_EXTRA
    echo "Built: $outdir/liblnn.so"
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2; exit 1;
    ;;
esac
