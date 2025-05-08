CC = gcc
MPICC = mpicc

CFLAGS = -O3 -Wall -std=c99
MPI_CFLAGS = -O3 -Wall -std=c99

# Default target
all: parallel

# Parallel executable
parallel: parallel.c
	$(MPICC) $(MPI_CFLAGS) -o parallel parallel.c -lm

# Clean up build artifacts
.PHONY: clean all
clean:
	rm -f parallel *.o *.out *.txt
