// Görkem Kadir Solun 22003214
// Usage: ./average-serial input_file output_file
// Reads a list of integers from the input file and calculates the average,
// then writes the result to output.txt.

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE* input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    int n;
    if (fscanf(input_file, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of elements\n");
        fclose(input_file);
        return EXIT_FAILURE;
    }

    int value;
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        if (fscanf(input_file, "%d", &value) != 1) {
            fprintf(stderr, "Error reading element %d\n", i);
            fclose(input_file);
            return EXIT_FAILURE;
        }
        sum += value;
    }

    fclose(input_file);

    double average = (double) sum / n;

    FILE* output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        perror("Error creating output file");
        return EXIT_FAILURE;
    }
    fprintf(output_file, "%f", average);
    fclose(output_file);

    return EXIT_SUCCESS;
}
