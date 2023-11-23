all: 
	nvcc -O3 -Xcompiler -fopenmp -o prog game_of_life.cu
	nvcc -O3 -Xcompiler -fopenmp -o prog2 warp_shuffle.cu

clean:
	rm -f prog
	rm -f prog2