#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel
__global__ void jacobi_kernel(double* A, double* b, double* x_old, double* x_new, int n);

// Main CUDA function
int jacobi_cuda(int n, double A[n][n], double b[n], double x[n], int max_iter, double tol, double *final_error){
    // GPU pointers
    double *d_A;
    double *d_b;
    double *d_x_old;
    double *d_x_new;

    // TODO:
    // cudaMalloc

    // TODO:
    // cudaMemcpy

    // TODO:
    // iterations

    // TODO:
    // copy result back

    // TODO:
    // free memory

    return max_iter;
}