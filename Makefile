all: prog

prog: main.cu
	nvcc -O3 -Xcompiler -fopenmp -o prog main.cu

clean:
	rm -f prog