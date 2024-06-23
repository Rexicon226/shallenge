NVCC=nvcc

all: main

main: main.cu sha256.cuh
	$(NVCC) -Xptxas -O3 -o $@ $< -ccbin clang-15 -lstdc++

clean:
	$(RM) $(filter-out main.cu helper.cuh sha256.cuh Makefile, $(wildcard *))

.PHONY: all, clean