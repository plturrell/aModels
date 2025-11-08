FC ?= gfortran
SRC := lnn_config.f90 lnn_profiler.f90 lnn_kernels.f90 tensor_kernels.f90 advanced_als.f90
OUTDIR := lib
CPPFLAGS_EXTRA := -cpp -fopenmp -march=native -ffast-math
LDFLAGS_EXTRA := -lgomp

ifdef LNN_USE_BLAS
  CPPFLAGS_EXTRA += -DUSE_BLAS -DLNN_USE_BLAS
  LDFLAGS_EXTRA += -lopenblas
endif

ifdef LNN_USE_LAPACK
  CPPFLAGS_EXTRA += -DUSE_LAPACK -DLNN_USE_LAPACK
  LDFLAGS_EXTRA += -llapack
endif

ifdef LNN_USE_DSYEVR
  CPPFLAGS_EXTRA += -DLNN_USE_DSYEVR
endif

ifdef LNN_USE_ACCELERATE
  CPPFLAGS_EXTRA += -DUSE_LAPACK -DLNN_USE_LAPACK
  LDFLAGS_EXTRA += -Wl,-framework,Accelerate
endif

ifdef LNN_PROFILE
  CPPFLAGS_EXTRA += -DLNN_PROFILE
endif

UNAME_S := $(shell uname -s)

all: lib

smart:
	@unameOut=`uname -s`; \
	if [ "$$unameOut" = "Darwin" ]; then \
		echo "[auto] Detected macOS; using Accelerate"; \
		$(MAKE) lib LNN_USE_ACCELERATE=1 FC=$(FC); \
	else \
		if ldconfig -p 2>/dev/null | grep -qi lapack || [ -e /usr/lib/x86_64-linux-gnu/liblapack.so ] || [ -e /usr/lib64/liblapack.so ]; then \
			echo "[auto] Detected LAPACK/OpenBLAS; enabling"; \
			$(MAKE) lib LNN_USE_LAPACK=1 LNN_USE_BLAS=1 FC=$(FC); \
		else \
			echo "[auto] No LAPACK; building default"; \
			$(MAKE) lib FC=$(FC); \
		fi; \
	fi

test: lib
	mkdir -p tests
	$(FC) -O2 -fopenmp -cpp -o tests/test_main tests/test_main.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	./tests/test_main
	$(FC) -O2 -fopenmp -cpp -o tests/test_edge_cases tests/test_edge_cases.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	./tests/test_edge_cases

lib:
	mkdir -p $(OUTDIR)
ifeq ($(UNAME_S),Darwin)
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_config.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_profiler.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_kernels.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) tensor_kernels.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) advanced_als.f90
	$(FC) -O3 -fPIC -dynamiclib -o $(OUTDIR)/liblnn_kernels.dylib lnn_kernels.o lnn_profiler.o $(LDFLAGS_EXTRA)
	$(FC) -O3 -fPIC -dynamiclib -o $(OUTDIR)/libtensor_kernels.dylib tensor_kernels.o advanced_als.o lnn_config.o $(LDFLAGS_EXTRA)
	$(FC) -O3 -fPIC -dynamiclib -o $(OUTDIR)/libadvanced_als.dylib advanced_als.o lnn_profiler.o lnn_config.o $(LDFLAGS_EXTRA)
	ar rcs $(OUTDIR)/liblnn_kernels.a lnn_kernels.o
	ar rcs $(OUTDIR)/libtensor_kernels.a tensor_kernels.o advanced_als.o lnn_profiler.o
	ar rcs $(OUTDIR)/libadvanced_als.a advanced_als.o lnn_profiler.o
else
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_config.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_profiler.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) lnn_kernels.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) tensor_kernels.f90
	$(FC) -O3 -fPIC -c $(CPPFLAGS_EXTRA) advanced_als.f90
	$(FC) -O3 -fPIC -shared -o $(OUTDIR)/liblnn_kernels.so lnn_kernels.o lnn_profiler.o $(LDFLAGS_EXTRA)
	$(FC) -O3 -fPIC -shared -o $(OUTDIR)/libtensor_kernels.so tensor_kernels.o advanced_als.o lnn_config.o $(LDFLAGS_EXTRA)
	$(FC) -O3 -fPIC -shared -o $(OUTDIR)/libadvanced_als.so advanced_als.o lnn_profiler.o lnn_config.o $(LDFLAGS_EXTRA)
	ar rcs $(OUTDIR)/liblnn_kernels.a lnn_kernels.o
	ar rcs $(OUTDIR)/libtensor_kernels.a tensor_kernels.o advanced_als.o lnn_profiler.o lnn_config.o
	ar rcs $(OUTDIR)/libadvanced_als.a advanced_als.o lnn_profiler.o lnn_config.o
endif

clean:
	rm -rf $(OUTDIR) *.o *.mod tests/test_main tests/test_edge_cases examples/cp_decomposition examples/spd_ops examples/topk_search

.PHONY: all lib clean test smart

bench: lib
	mkdir -p bench
	$(FC) -O2 -fopenmp -cpp -o bench/bench_main bench/bench_main.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	@echo "Run: ./bench/bench_main"

examples: lib
	$(FC) -O2 -fopenmp -cpp -o examples/cp_decomposition examples/cp_decomposition.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	$(FC) -O2 -fopenmp -cpp -o examples/spd_ops examples/spd_ops.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	$(FC) -O2 -fopenmp -cpp -o examples/topk_search examples/topk_search.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	@echo "Built examples: examples/cp_decomposition, examples/spd_ops, examples/topk_search"

test_lapack:
	$(MAKE) clean
	$(MAKE) lib FC=$(FC) LNN_USE_LAPACK=1 LNN_USE_BLAS=1
	mkdir -p tests
	$(FC) -O2 -fopenmp -cpp -o tests/test_main tests/test_main.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	./tests/test_main
	$(FC) -O2 -fopenmp -cpp -o tests/test_edge_cases tests/test_edge_cases.f90 lnn_config.o lnn_kernels.o tensor_kernels.o advanced_als.o $(LDFLAGS_EXTRA)
	./tests/test_edge_cases
