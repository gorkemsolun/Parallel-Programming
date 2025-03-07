// Görkem Kadir Solun 22003214
// Usage: ./bucket_average-serial input_file output_file
// Reads a list of key-value pairs from the input file and calculates the average value for each key.
// The output file will contain the average values for each key in ascending order.

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return 1;
    }

    FILE* input_file = fopen(argv[1], "r");
    if (!input_file) {
        perror("Error opening input file");
        return 1;
    }

    int n;
    if (fscanf(input_file, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of pairs\n");
        fclose(input_file);
        return 1;
    }

    int* keys = (int*) malloc(n * sizeof(int));
    int* values = (int*) malloc(n * sizeof(int));
    if (!keys || !values) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        if (fscanf(input_file, "%d %d", &keys[i], &values[i]) != 2) {
            fprintf(stderr, "Error reading pair %d\n", i);
            free(keys); free(values);
            fclose(input_file);
            return 1;
        }
    }
    fclose(input_file);

    // Determine the minimum and maximum key values to determine the bucket count.
    int min_key = keys[0], max_key = keys[0];
    for (int i = 1; i < n; i++) {
        if (keys[i] < min_key) {
            min_key = keys[i];
        }
        if (keys[i] > max_key) {
            max_key = keys[i];
        }
    }
    int bucket_count = max_key - min_key + 1;

    // Allocate arrays for bucket sums and counts.
    double* bucket_sums = (double*) calloc(bucket_count, sizeof(double));
    int* bucket_counts = (int*) calloc(bucket_count, sizeof(int));
    if (!bucket_sums || !bucket_counts) {
        fprintf(stderr, "Memory allocation failed for buckets\n");
        free(keys); free(values);
        return 1;
    }

    // Accumulate sums and counts in the proper bucket.
    for (int i = 0; i < n; i++) {
        int index = keys[i] - min_key;
        bucket_sums[index] += values[i];
        bucket_counts[index] += 1;
    }

    // Write the averages to the output file.
    FILE* output_file = fopen(argv[2], "w");
    if (!output_file) {
        perror("Error opening output file");
        free(keys);
        free(values);
        free(bucket_sums);
        free(bucket_counts);
        return 1;
    }

    for (int j = 0; j < bucket_count; j++) {
        if (bucket_counts[j] > 0) {
            double average = bucket_sums[j] / bucket_counts[j];
            fprintf(output_file, "%f\n", average);
        }
    }
    fclose(output_file);

    free(keys);
    free(values);
    free(bucket_sums);
    free(bucket_counts);
    return 0;
}
