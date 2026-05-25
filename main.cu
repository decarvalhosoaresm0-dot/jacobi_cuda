#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define N 1000
#define SEED 42

// function declaration
int jacobi_cuda(int n, double *A, double *b, double *x, int max_iter, double tol, double *final_error);

// generates a system
void generate_system(int n, double *A, double *b, double *x){

    srand(SEED);

    for (int i = 0; i < n; i++) {

        double row_sum = 0.0;

        for (int j = 0; j < n; j++) {
            if(i != j){

                A[i * n + j] = ((double) rand() / RAND_MAX) * 10.0; // random value between 0 and 10
                row_sum += fabs(A[i * n + j]);
            }
        }

        A[i * n +i] = row_sum + 10.0;

        b[i] = 1.0;
        x[i] = 0.0;
    }
}

// prints the system with a limit of N <= 5
void print_system(int n, double *A, double *b){

    int limit = (n < 5) ? n : 5;

    printf("\nSystem Ax = b (showing first %d rows):\n\n", limit);

    for(int i = 0; i < limit; i++){

        // matrix A
        printf("[ ");
        for(int j = 0; j < limit; j++){
            printf("%6.2f ", A[i * n + j]);
        }

        if(n > limit) printf("... ");
        printf("] ");

        // xN
        printf("[x%d]", i+1);

        // sinal =
        if(i == limit/2)
            printf(" = ");
        else
            printf("   ");

        // vector b
        printf("[ %6.2f ]", b[i]);

        printf("\n");
    }

    // cut indication
    if(n > limit){
        printf("  ...\n");
    }

    printf("\n");
}

int main()
{
    static double A[N * N];
    static double b[N];
    static double x[N];

    int max_iter = 5000;
    double tol = 1e-3;
    double final_error;

    generate_system(N, A, b, x);

    print_system(N, A, b);

    clock_t start = clock();
    int iterations = jacobi_cuda(N, A, b, x, max_iter, tol, &final_error);

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("Iterations: %d\n", iterations);
    printf("Final error: %e\n", final_error);
    printf("Elapsed time: %f seconds\n", elapsed);

    return 0;
}