// Görkem Kadir Solun 22003214
// Usage: mpirun -np number_of_processes ./bucket_average-mpi input_file output_file
// Reads a list of key-value pairs from the input file and calculates the average value for each key.


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

typedef struct {
    int key;
    int value;
} Pair;

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time, local_time, max_time;

    MPI_Init(&argc, &argv);
    start_time = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int n;
    int min_key, max_key, bucket_count;
    Pair* pairs = NULL;

    if (rank == 0) {
        FILE* input_file = fopen(argv[1], "r");
        if (!input_file) {
            perror("Error opening input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fscanf(input_file, "%d", &n) != 1) {
            fprintf(stderr, "Error reading number of pairs\n");
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        pairs = (Pair*) malloc(n * sizeof(Pair));
        if (!pairs) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < n; i++) {
            if (fscanf(input_file, "%d %d", &pairs[i].key, &pairs[i].value) != 2) {
                fprintf(stderr, "Error reading pair %d\n", i);
                free(pairs);
                fclose(input_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(input_file);

        // Determine the overall key range to determine the bucket count
        min_key = pairs[0].key;
        max_key = pairs[0].key;
        for (int i = 1; i < n; i++) {
            if (pairs[i].key < min_key) {
                min_key = pairs[i].key;
            }
            if (pairs[i].key > max_key) {
                max_key = pairs[i].key;
            }
        }
        bucket_count = max_key - min_key + 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_key, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bucket_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide the work equally
    int local_n = n / size;
    Pair* local_pairs = (Pair*) malloc(local_n * sizeof(Pair));
    if (!local_pairs) {
        fprintf(stderr, "Process %d: Memory allocation failed for local pairs\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Define an MPI datatype for the Pair structure
    MPI_Datatype MPI_PAIR;
    MPI_Type_contiguous(2, MPI_INT, &MPI_PAIR);
    MPI_Type_commit(&MPI_PAIR);

    // Scatter the pairs from the master to all processes.
    MPI_Scatter(pairs, local_n, MPI_PAIR, local_pairs, local_n, MPI_PAIR, 0, MPI_COMM_WORLD);
    // free the master copy after scattering
    if (rank == 0) {
        free(pairs);
    }

    // Each process allocates local arrays for bucket sums and counts.
    double* local_bucket_sums = (double*) calloc(bucket_count, sizeof(double));
    int* local_bucket_counts = (int*) calloc(bucket_count, sizeof(int));
    if (!local_bucket_sums || !local_bucket_counts) {
        fprintf(stderr, "Process %d: Memory allocation failed for buckets\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < local_n; i++) {
        int bucket_index = local_pairs[i].key - min_key;
        local_bucket_sums[bucket_index] += local_pairs[i].value;
        local_bucket_counts[bucket_index] += 1;
    }
    free(local_pairs);

    // Prepare arrays on the master to gather the global sums and counts.
    double* global_bucket_sums = NULL;
    int* global_bucket_counts = NULL;
    if (rank == 0) {
        global_bucket_sums = (double*) calloc(bucket_count, sizeof(double));
        global_bucket_counts = (int*) calloc(bucket_count, sizeof(int));
        if (!global_bucket_sums || !global_bucket_counts) {
            fprintf(stderr, "Master: Memory allocation failed for global buckets\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Reduce(local_bucket_sums, global_bucket_sums, bucket_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_bucket_counts, global_bucket_counts, bucket_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    free(local_bucket_sums);
    free(local_bucket_counts);

    // The master computes the averages and writes the output file.
    if (rank == 0) {
        FILE* output_file = fopen(argv[2], "w");
        if (!output_file) {
            perror("Error opening output file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int j = 0; j < bucket_count; j++) {
            if (global_bucket_counts[j] > 0) {
                double average = global_bucket_sums[j] / global_bucket_counts[j];
                fprintf(output_file, "%f\n", average);
            }
        }
        fclose(output_file);
        free(global_bucket_sums);
        free(global_bucket_counts);
    }

    MPI_Type_free(&MPI_PAIR);
    end_time = MPI_Wtime();
    local_time = end_time - start_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Longest running process time for bucket average mpi: %f, input count: %d, process count: %d\n", max_time, n, size);
    }

    MPI_Finalize();
    return 0;
}
