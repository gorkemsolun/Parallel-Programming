// Görkem Kadir Solun 
// Serial Conjugate Gradient solver for Ax = b
// Reads matrix A and vector b from input files, solves for x, and writes x to s_output.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [output_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_file_name = argv[1];
    const char* outputFileName = (argc == 3) ? argv[2] : "s_output.txt";
    FILE* file = fopen(input_file_name, "r");
    if (!file) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    int N;
    // First value: dimension
    if (fscanf(file, "%d", &N) != 1) {
        fprintf(stderr, "Failed to read dimension\n");
        return EXIT_FAILURE;
    }

    // Allocate arrays
    double* A = malloc(N * N * sizeof(double));
    double* b = malloc(N * sizeof(double));
    double* x = calloc(N, sizeof(double));  // x=0
    double* r = malloc(N * sizeof(double));
    double* p = malloc(N * sizeof(double));
    double* Ap = malloc(N * sizeof(double));
    if (!A || !b || !x || !r || !p || !Ap) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Read matrix entries (row-major)
    for (int i = 0; i < N * N; ++i) {
        if (fscanf(file, "%lf", &A[i]) != 1) {
            fprintf(stderr, "Failed to read A[%d]\n", i);
            return EXIT_FAILURE;
        }
    }

    // Read vector b
    for (int i = 0; i < N; ++i) {
        if (fscanf(file, "%lf", &b[i]) != 1) {
            fprintf(stderr, "Failed to read b[%d]\n", i);
            return EXIT_FAILURE;
        }
    }
    fclose(file);

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
        if (sqrt(rhoNew) < 1e-6) {
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

    FILE* outputFile = fopen(outputFileName, "w");
    if (!outputFile) {
        perror("Error opening output file");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        fprintf(outputFile, "%lf\n", x[i]);
    }

    fprintf(outputFile, "Elapsed time: %f seconds\n", elapsed);

    fclose(outputFile);

    free(A);
    free(b);
    free(x);
    free(r);
    free(p);
    free(Ap);

    return EXIT_SUCCESS;
}
