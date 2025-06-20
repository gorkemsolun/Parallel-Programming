// Görkem Kadir Solun 22003214
// Usage: mpirun -np number_of_processes ./average-mpi-ppv1 input_file output_file
// Reads a list of integers from the input file and calculates the average.


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();  // Start timing

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n;
    int* data = NULL;

    // Master reads input file and sends n to other processes.
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
        for (int process = 1; process < size; process++) {
            MPI_Send(&n, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int base = n / size;
    int remainder = n % size;
    int local_n = (rank < remainder) ? base + 1 : base;

    int* local_data = (int*) malloc(local_n * sizeof(int));
    if (!local_data) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < local_n; i++) {
            local_data[i] = data[offset++];
        }
        for (int process = 1; process < size; process++) {
            int proc_n = (process < remainder) ? base + 1 : base;
            MPI_Send(&data[offset], proc_n, MPI_INT, process, 0, MPI_COMM_WORLD);
            offset += proc_n;
        }
    } else {
        MPI_Recv(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    long long local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    if (rank == 0) {
        long long total_sum = local_sum;
        long long received_sum;
        for (int process = 1; process < size; process++) {
            MPI_Recv(&received_sum, 1, MPI_LONG_LONG, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_sum;
        }
        double average = (double) total_sum / n;

        FILE* output_file = fopen(argv[2], "w");
        if (!output_file) {
            perror("Error opening output file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(output_file, "%f", average);
        fclose(output_file);
    } else {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    free(local_data);
    if (rank == 0) {
        free(data);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();  // End timing
    double local_elapsed = end_time - start_time;
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Longest process time for average mpi ppv1: %f seconds, input size: %d, process count: %d\n", max_elapsed, n, size);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
