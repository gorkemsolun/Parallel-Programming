#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE* file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    int n;
    if (fscanf(file, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of elements\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    int value;
    long long sum = 0;
    for (int i = 0; i < n; i++) {   
        if (fscanf(file, "%d", &value) != 1) {
            fprintf(stderr, "Error reading element %d\n", i);
            fclose(file);
            return EXIT_FAILURE;
        }
        sum += value;
    }

    fclose(file);

    double average = (double) sum / n;
    printf("Average: %f\n", average);

    return EXIT_SUCCESS;
}
