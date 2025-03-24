// Görkem Kadir Solun 22003214
// Usage: mpirun -np number_of_processes ./average-mpi-ppv2 input_file output_file
// Reads a list of integers from the input file and calculates the average.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n;
    int* data = NULL;

    if (rank == 0) {
        FILE* input_file = fopen(argv[1], "r");
        if (!input_file) {
            perror("Error opening input file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (fscanf(input_file, "%d", &n) != 1) {
            fprintf(stderr, "Error reading number of elements\n");
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (n % size != 0) {
            fprintf(stderr, "Error: %d elements not divisible by %d processes.\n", n, size);
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        data = (int*) malloc(n * sizeof(int));
        if (!data) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < n; i++) {
            if (fscanf(input_file, "%d", &data[i]) != 1) {
                fprintf(stderr, "Error reading element %d\n", i);
                fclose(input_file);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        fclose(input_file);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    int* local_data = (int*) malloc(local_n * sizeof(int));
    if (!local_data) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    long long global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* output_file = fopen(argv[2], "w");
        if (!output_file) {
            perror("Error opening output file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        double average = (double) global_sum / n;
        fprintf(output_file, "%f", average);
        fclose(output_file);
    }

    free(local_data);
    if (rank == 0) {
        free(data);
    }

    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Longest run time for average mpi ppv2: %f seconds, input size: %d, process count: %d\n",
            max_elapsed, n, size);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
