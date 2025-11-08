#!/usr/bin/env bash
set -euo pipefail

# Build helper for Linux x86_64 (KVM), targeting OpenBLAS/LAPACK.
# Usage examples:
#   ./scripts/build_linux.sh            # no BLAS/LAPACK
#   LNN_USE_BLAS=1 LNN_USE_LAPACK=1 ./scripts/build_linux.sh
#   LNN_USE_DSYEVR=1 LNN_USE_BLAS=1 LNN_USE_LAPACK=1 ./scripts/build_linux.sh

FC=${FC:-gfortran}
MAKE=${MAKE:-make}

# Detect CPU and choose conservative flags if under KVM
MARCH=${MARCH:-}
if [[ -z "$MARCH" ]]; then
  if grep -qi kvm /proc/cpuinfo 2>/dev/null; then
    # Safe baseline for most x86_64 servers; adjust if you know AVX-512 is available.
    MARCH="-march=x86-64-v3 -mtune=generic"
  else
    MARCH="-march=native -mtune=native"
  fi
fi
export CPPFLAGS_EXTRA="-cpp -fopenmp ${CPPFLAGS_EXTRA:-} ${MARCH}"

if [[ "${LNN_USE_BLAS:-}" == "1" ]]; then
  export LNN_USE_BLAS=1
fi
if [[ "${LNN_USE_LAPACK:-}" == "1" ]]; then
  export LNN_USE_LAPACK=1
fi
if [[ "${LNN_USE_DSYEVR:-}" == "1" ]]; then
  export LNN_USE_DSYEVR=1
fi

echo "Building with FC=$FC MARCH='$MARCH' BLAS=${LNN_USE_BLAS:-0} LAPACK=${LNN_USE_LAPACK:-0} DSYEVR=${LNN_USE_DSYEVR:-0}"

$MAKE clean
$MAKE lib FC="$FC"
echo "Done. Libraries in ./lib"

