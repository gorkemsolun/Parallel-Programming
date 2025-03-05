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

    // Master process reads input file and sends n to other processes.
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
        // Send n to each worker.
        for (int proc = 1; proc < size; proc++) {
            MPI_Send(&n, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
    } else {
        // Workers receive n from the master.
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Determine local chunk sizes (distributing as evenly as possible)
    int base = n / size;
    int rem = n % size;
    int local_n = (rank < rem) ? base + 1 : base;

    int* local_data = (int*) malloc(local_n * sizeof(int));
    if (local_data == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        int offset = 0;
        // Rank 0 copies its own chunk.
        for (int i = 0; i < local_n; i++) {
            local_data[i] = data[offset++];
        }
        // Send chunks to each worker.
        for (int proc = 1; proc < size; proc++) {
            int proc_n = (proc < rem) ? base + 1 : base;
            MPI_Send(&data[offset], proc_n, MPI_INT, proc, 0, MPI_COMM_WORLD);
            offset += proc_n;
        }
    } else {
        // Workers receive their chunk.
        MPI_Recv(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each process computes its local sum.
    long long local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    if (rank == 0) {
        // Master collects local sums from workers.
        long long total_sum = local_sum;
        long long recv_sum;
        for (int proc = 1; proc < size; proc++) {
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }
        double average = (double) total_sum / n;
        printf("Average: %f\n", average);
    } else {
        // Workers send their local sum to the master.
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    free(local_data);
    if (rank == 0)
        free(data);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
