NVCC=nvcc

all: main

main: main.cu sha256.cuh
	$(NVCC) -Xptxas -O3,-v -o $@ $< -ccbin clang-15 -lstdc++

clean:
	rm -f main

.PHONY: all, clean