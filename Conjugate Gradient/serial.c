// Görkem Kadir Solun 
// Serial Conjugate Gradient solver for Ax = b
// Reads matrix A and vector b from input files, solves for x, and writes x to s_output.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_file> <vector_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* matrixFile = argv[1];
    const char* vecfile = argv[2];

    FILE* fileA = fopen(matrixFile, "r");
    if (!fileA) {
        perror("Error opening matrix file");
        return EXIT_FAILURE;
    }
    FILE* fileb = fopen(vecfile, "r");
    if (!fileb) {
        perror("Error opening vector file");
        fclose(fileA);
        return EXIT_FAILURE;
    }

    int N;
    // First value: dimension
    if (fscanf(fileA, "%d", &N) != 1) {
        fprintf(stderr, "Failed to read matrix dimension\n");
        return EXIT_FAILURE;
    }
    int Nb;
    if (fscanf(fileb, "%d", &Nb) != 1 || Nb != N) {
        fprintf(stderr, "Vector dimension does not match or failed to read\n");
        return EXIT_FAILURE;
    }

    // Allocate arrays
    double* A = malloc(N * N * sizeof(double));
    double* b = malloc(N * sizeof(double));
    double* x = calloc(N, sizeof(double));  // Initial guess x=0
    double* r = malloc(N * sizeof(double));
    double* p = malloc(N * sizeof(double));
    double* Ap = malloc(N * sizeof(double));
    if (!A || !b || !x || !r || !p || !Ap) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Read matrix entries (row-major)
    for (int i = 0; i < N * N; ++i) {
        if (fscanf(fileA, "%lf", &A[i]) != 1) {
            fprintf(stderr, "Failed to read A[%d]\n", i);
            return EXIT_FAILURE;
        }
    }
    fclose(fileA);

    // Read vector b
    for (int i = 0; i < N; ++i) {
        if (fscanf(fileb, "%lf", &b[i]) != 1) {
            fprintf(stderr, "Failed to read b[%d]\n", i);
            return EXIT_FAILURE;
        }
    }
    fclose(fileb);

    // Conjugate Gradient variables
    double epsilon = 1e-6;
    double alpha, beta, rho;

    // r = b - A*x (x=0 to r=b)
    for (int i = 0; i < N; ++i) {
        r[i] = b[i];
        p[i] = r[i];
    }
    rho = 0;
    for (int i = 0; i < N; ++i) {
        rho += r[i] * r[i];
    }

    clock_t t_start = clock();

    // Max iterations is N to avoid infinite loop
    for (int k = 0; k < N; ++k) {
        // Ap = A * p
        for (int i = 0; i < N; ++i) {
            Ap[i] = 0.0;
            for (int j = 0; j < N; ++j) {
                Ap[i] += A[i * N + j] * p[j];
            }
        }

        // alpha = rho / (p^T * Ap)   //   (i.e., pi = <p, Ap> and alpha = rho / pi)
        double pAp = 0;
        for (int i = 0; i < N; ++i) {
            pAp += p[i] * Ap[i];
        }
        alpha = rho / pAp;

        // x = x + alpha * p
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * p[i];
        }
        // r = r - alpha * Ap
        for (int i = 0; i < N; ++i) {
            r[i] -= alpha * Ap[i];
        }

        // convergence
        double rhoNew = 0;
        for (int i = 0; i < N; ++i) {
            rhoNew += r[i] * r[i];
        }
        if (sqrt(rhoNew) < epsilon) {
            break;
        }

        // p = r + (rhoNew/rho) * p
        beta = rhoNew / rho;
        for (int i = 0; i < N; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rho = rhoNew;
    }

    clock_t t_end = clock();
    double elapsed = (double) (t_end - t_start) / CLOCKS_PER_SEC;
    fprintf(stderr, "Elapsed time: %f seconds\n", elapsed);

    FILE* fx = fopen("s_output.txt", "w");
    if (!fx) {
        perror("Error opening output file");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        fprintf(fx, "%lf\n", x[i]);
    }
    fclose(fx);

    free(A);
    free(b);
    free(x);
    free(r);
    free(p);
    free(Ap);

    return EXIT_SUCCESS;
}
