    #include <cstdio>
    #include <cstdlib>
    #include <omp.h>
    #include <cuda.h>
    #include <iostream>
    #include <string>
    using namespace std;
    // Tamaño de bloque
    #define BSIZE 32
    #define BSIZE1D 1024

    __global__ void gpu_sim(bool *M1, int n, bool *M2);
    // __global__ void kernel2(bool *M1, int n, bool *M2);
    void inicializa_matrix(bool *M1, int n, int seed);
    void printM(bool *M1, int n);
    void copy(bool *M1, int n, bool *M2);
    void cpu_sim(bool *M1, int n, bool *M2);



    int main(int argc, char **argv){
        // Se 
        if(argc != 6){
            fprintf(stderr, "run as ./prog <gpu-id>  n seed pasos <block-size>\n");
            exit(EXIT_FAILURE);
        }
        // Inicializacion 
        printf("Inicializando.....\n");
        string str = "";
        int gpu_id = atoi(argv[1]); // ID de la gpu
        int n = atoi(argv[2]); // Size problem [nxN]
        int seed = atoi(argv[3]); // Seed 
        int steps = atoi(argv[4]); // Steps number
        int nb = atoi(argv[5]);
        printf("<gpu_id: %d>, <n: %d> , <seed: %d> , <steps: %d>, <nt: %d>, <nb: %d>, <mode: %d>\n", gpu_id, n, seed, steps, nb);
        bool * M1 = new bool[n*n];
        bool * M2 = new bool[n*n];
        inicializa_matrix(M1, n, seed);
        if( n <= 50){
            printf("Matrix inicial: \n");
            printM(M1,n);
        }

        printf("Inicializando variables para GPU\n");
        float msecs = 0.0f;
        bool * dx, * dy;
        // Se selecciona la gpu y se muestra
        cudaSetDevice(gpu_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_id);
        printf("GPU: %s\n", prop.name);
        // Se ingresan las matrices en la memoria de la GPU 
        cudaMalloc(&dx, sizeof(bool) * (n * n));
        cudaMalloc(&dy, sizeof(bool) * (n * n));
        cudaMemcpy(dx, M1, sizeof(bool) * (n * n), cudaMemcpyHostToDevice);
        // Se crean los eventos para medir el tiempo
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // Se eligen los tamaños de block y grid 
        cudaEventRecord(start);
        dim3 block(BSIZE, BSIZE, 1);
        dim3 grid ((n + (BSIZE - 1))/BSIZE, (n + (BSIZE - 1))/BSIZE, 1);
        if(nb != 0){
            block = dim3(BSIZE, 1, 1);
            grid = dim3(nb, 1, 1);
        }
        while(str == "" && steps != 0){
            gpu_sim<<<grid, block>>>(dx, n, dy);
            if(n <= 128){
                cudaMemcpy(M2, dy, sizeof(bool)*(n*n), cudaMemcpyDeviceToHost);
                printf("------------\n");
                printM(M2,n);
                cout << "Para hacer otra generacion <enter>, para terminar cualquier cosa";
                getline(cin, str);
            }
            // Se sincroniza
            cudaDeviceSynchronize();
            cudaMemcpy(dx, dy, sizeof(bool) * (n * n), cudaMemcpyDeviceToDevice);
            steps--;
        }    
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        // Se calcula el tiempo
        cudaEventElapsedTime(&msecs, start, stop);
        //Se traspasa la informacion al host

        cudaMemcpy(M2, dy, sizeof(bool)*(n*n), cudaMemcpyDeviceToHost);
        printf("done: time gpu: %f secs\n", msecs/1000.0f);
        //Se libera la memoria de la GPU
        cudaFree(dx);
        cudaFree(dy);
        
        // Se libera la memoria utilizada
        delete M1;
        delete M2;
        printf("El programa termino con exito!\n");
        fflush(stdout);
        exit(EXIT_SUCCESS);
    }


    /* Funciones auxiliares */

    void copy(bool *M1, int n, bool *M2){
        for (int i = 0; i < n * n; i++){
                M1[i] = M2[i];
        }
    }

    void inicializa_matrix(bool *M1, int n, int seed){
        srand(seed);
        for(int i = 0; i < n * n; ++i){
            M1[i] = rand()%2;
        }
    }
    
    void printM(bool *M1, int n){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                if(M1[i * n + j]) printf("1 ");
                else printf("0 ");
            }
            printf("\n");
        }
    }
        /* Kernel principal*/

    __global__ void gpu_sim(bool *M1, int n, bool *M2){
        // Numero de celulas vivas alrededor
        int neighbour_live_cell = 0;
        int tidx = blockIdx.x * blockDim.x  +  threadIdx.x;
        int tidy = blockIdx.y * blockDim.y +  threadIdx.y;
        int pos = (tidy * n) + tidx;
        if(tidx < n && tidy < n){
            for (int i = tidx - 1; i <= tidx + 1; i++) {
                for (int j = tidy - 1; j <= tidy + 1; j++) {
                    if ((i == tidx && j == tidy) || (i < 0 || j < 0)
                        || (i >= n || j >= n)) continue;
                    if (M1[j*n + i]) neighbour_live_cell++;
                }
            }
            if (M1[pos] && (neighbour_live_cell == 2 
                || neighbour_live_cell == 3)) M2[pos] = 1;
            else if (!(M1[pos]) && neighbour_live_cell == 3) M2[pos] = 1;
            else M2[pos] = 0;
        }

    }