#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n; // total number of elements
    int* data = NULL;

    if (rank == 0) {
        FILE* file = fopen(argv[1], "r");
        if (!file) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (fscanf(file, "%d", &n) != 1) {
            fprintf(stderr, "Error reading number of elements\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // Check that n is divisible by the number of processes.
        if (n % size != 0) {
            fprintf(stderr, "Error: number of elements (%d) is not divisible by number of processes (%d).\n", n, size);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        data = (int*) malloc(n * sizeof(int));
        if (data == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < n; i++) {
            if (fscanf(file, "%d", &data[i]) != 1) {
                fprintf(stderr, "Error reading element %d\n", i);
                fclose(file);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        fclose(file);
    }

    // Broadcast n so that every process knows the total count.
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // With n divisible by size, determine local data size.
    int local_n = n / size;
    int* local_data = (int*) malloc(local_n * sizeof(int));
    if (local_data == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Scatter the data from the master to all processes.
    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its local sum.
    long long local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    // Reduce all local sums to the master.
    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double average = (double) global_sum / n;
        printf("Average: %f\n", average);
    }

    free(local_data);
    if (rank == 0)
        free(data);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
