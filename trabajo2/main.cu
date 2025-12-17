#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>
#include <cstdlib>
#include <cmath>

using namespace std;

// ============================================================
// 1. Versión CPU multicore (OpenMP)
// ============================================================
void matmulCPU(float* A, float* B, float* C, int n, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float suma = 0.0f;
            for (int k = 0; k < n; k++) {
                suma += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = suma;
        }
    }
}

// ============================================================
// 2. Versión GPU básica
// ============================================================
__global__ void matmulGPU(float* A, float* B, float* C, int n) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < n && col < n) {
        float suma = 0.0f;
        for (int k = 0; k < n; k++) {
            suma += A[fila*n + k] * B[k*n + col];
        }
        C[fila*n + col] = suma;
    }
}

// ============================================================
// 3. Versión GPU con memoria compartida
// ============================================================
#define TILE_SIZE 16

__global__ void matmulGPUSm(float* A, float* B, float* C, int n) {
    __shared__ float bloqueA[TILE_SIZE][TILE_SIZE];
    __shared__ float bloqueB[TILE_SIZE][TILE_SIZE];

    int fila = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col  = blockIdx.x * TILE_SIZE + threadIdx.x;

    float suma = 0.0f;

    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; m++) {

        if (fila < n && m*TILE_SIZE + threadIdx.x < n)
            bloqueA[threadIdx.y][threadIdx.x] =
                A[fila*n + m*TILE_SIZE + threadIdx.x];
        else
            bloqueA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && m*TILE_SIZE + threadIdx.y < n)
            bloqueB[threadIdx.y][threadIdx.x] =
                B[(m*TILE_SIZE + threadIdx.y)*n + col];
        else
            bloqueB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            suma += bloqueA[threadIdx.y][k] * bloqueB[k][threadIdx.x];

        __syncthreads();
    }

    if (fila < n && col < n)
        C[fila*n + col] = suma;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {

    if (argc < 4) {
        cout << "Uso: ./prog <n> <num_threads_cpu> <ALG>\n";
        cout << "ALG = 1 (CPU), 2 (GPU), 3 (GPU Shared)\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads_cpu = atoi(argv[2]);
    int alg = atoi(argv[3]);

    size_t tam = n * n * sizeof(float);

    // Reservar memoria en CPU
    float *h_A = new float[n*n];
    float *h_B = new float[n*n];
    float *h_C = new float[n*n];

    // Inicializar matrices
    srand(0);
    for (int i = 0; i < n*n; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ===================== CPU =====================
    if (alg == 1) {

        auto inicio = chrono::high_resolution_clock::now();
        matmulCPU(h_A, h_B, h_C, n, num_threads_cpu);
        auto fin = chrono::high_resolution_clock::now();

        chrono::duration<double> tiempo = fin - inicio;
        cout << "Tiempo CPU: " << tiempo.count() << " s\n";
    }

    // ===================== GPU =====================
    else {

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, tam);
        cudaMalloc(&d_B, tam);
        cudaMalloc(&d_C, tam);

        cudaMemcpy(d_A, h_A, tam, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, tam, cudaMemcpyHostToDevice);

        dim3 hilos(TILE_SIZE, TILE_SIZE);
        dim3 bloques((n + TILE_SIZE - 1) / TILE_SIZE,
                      (n + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t inicio, fin;
        cudaEventCreate(&inicio);
        cudaEventCreate(&fin);

        cudaEventRecord(inicio);

        if (alg == 2)
            matmulGPU<<<bloques, hilos>>>(d_A, d_B, d_C, n);
        else if (alg == 3)
            matmulGPUSm<<<bloques, hilos>>>(d_A, d_B, d_C, n);

        cudaEventRecord(fin);
        cudaEventSynchronize(fin);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, inicio, fin);

        cout << "Tiempo GPU: " << ms / 1000.0 << " s\n";

        cudaMemcpy(h_C, d_C, tam, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(inicio);
        cudaEventDestroy(fin);
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
