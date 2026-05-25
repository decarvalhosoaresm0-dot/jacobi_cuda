#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//Verificacao de erros nas chamadas CUDA
#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__);       \
            fprintf(stderr, "Error code: %d\n", (int)err);                     \
            fprintf(stderr, "Error name: %s\n", cudaGetErrorName(err));        \
            fprintf(stderr, "Error string: %s\n", cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                      \
    } while (0)

// CUDA kernel
__global__ void jacobi_kernel(double* A, double* b, double* x_old, double* x_new, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            if (j != i) {
                sum += A[i * n + j] * x_old[j];
            }
        }

        x_new[i] = (b[i] - sum) / A[i * n + i];
    }
}

// Kernel para calcular para calcular a soma dos quadrados dos erros por bloco.
// Depois fazemos uma reducao dentro do bloco em memoria compartilhada --> mais rapido que memoria global.
__global__ void error_kernel(double* x_old, double* x_new, double* block_errors, int n)
{
    extern __shared__ double sdata[];

    int thr_id = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_error = 0.0;

    if (i < n) {
        double diff = x_new[i] - x_old[i];
        local_error = diff * diff;
    }

    sdata[thr_id] = local_error;
    __syncthreads();

    // Reducao para SOMAR os erros quadráticos dentro do bloco
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thr_id < stride) {
            sdata[thr_id] += sdata[thr_id + stride];
        }
        __syncthreads();
    }

    // A thread 0 salva a soma dos quadrados daquele bloco
    if (thr_id == 0) {
        block_errors[blockIdx.x] = sdata[0];
    }
}

// Main CUDA function
int jacobi_cuda(int n, double *A, double *b, double *x, int max_iter, double tol, double *final_error){
    // GPU pointers
    double *d_A = NULL;
    double *d_b = NULL;
    double *d_x_old = NULL;
    double *d_x_new = NULL;
    double *d_block_errors = NULL;

    int iterations;
    double error = 0.0;

    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    size_t matrix_size = (size_t)n * (size_t)n * sizeof(double);
    size_t vector_size = (size_t)n * sizeof(double);
    size_t error_size = (size_t)blocks_per_grid * sizeof(double);

    double *h_block_errors = (double*)malloc(error_size);

    if (h_block_errors == NULL) {
        fprintf(stderr, "Error: could not allocate host memory for block errors.\n");
        exit(EXIT_FAILURE);
    }

    // cudaMalloc
    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, vector_size));
    CUDA_CHECK(cudaMalloc((void**)&d_x_old, vector_size));
    CUDA_CHECK(cudaMalloc((void**)&d_x_new, vector_size));
    CUDA_CHECK(cudaMalloc((void**)&d_block_errors, error_size));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(d_A, A, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, vector_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_old, x, vector_size, cudaMemcpyHostToDevice));

    // iterations
    for (iterations = 0; iterations < max_iter; iterations++) {
        jacobi_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_b, d_x_old, d_x_new, n);
        CUDA_CHECK(cudaGetLastError());

        error_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(double)>>>(d_x_old, d_x_new, d_block_errors, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_block_errors, d_block_errors, error_size, cudaMemcpyDeviceToHost));

        double sum_error_squared = 0.0;

        // h_block_errors[i] guarda a soma dos quadrados dos erros daquele bloco.
        for (int i = 0; i < blocks_per_grid; i++) {
            sum_error_squared += h_block_errors[i];
        }

        // Norma euclidiana do vetor de erro
        error = sqrt(sum_error_squared);

        if (error < tol) {
            iterations++;
            CUDA_CHECK(cudaMemcpy(x, d_x_new, vector_size, cudaMemcpyDeviceToHost));
            break;
        }

        // troca x_new e x_old para a proxima iteracao
        double *temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;
    }

    // Caso tenha parado por max_iter, o vetor mais recente esta em d_x_old.
    if (iterations == max_iter) {
        CUDA_CHECK(cudaMemcpy(x, d_x_old, vector_size, cudaMemcpyDeviceToHost));
    }

    if (final_error != NULL) {
        *final_error = error;
    }

    // free memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x_old));
    CUDA_CHECK(cudaFree(d_x_new));
    CUDA_CHECK(cudaFree(d_block_errors));
    free(h_block_errors);

    return iterations;
}