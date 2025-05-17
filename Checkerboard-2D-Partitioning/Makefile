CC = gcc
MPICC = mpicc

CFLAGS = -O3 -Wall -std=c99
MPI_CFLAGS = -O3 -Wall -std=c99

# Default target
all: parallel row_parallel

# Parallel executable
parallel: parallel.c
	$(MPICC) $(MPI_CFLAGS) -o parallel parallel.c -lm

row_parallel: row_parallel.c
	$(MPICC) $(MPI_CFLAGS) -o row_parallel row_parallel.c -lm

# Clean up build artifacts
.PHONY: clean all
clean:
	rm -f parallel *.o *.out *.txt row_parallel *.o *.out *.txt

