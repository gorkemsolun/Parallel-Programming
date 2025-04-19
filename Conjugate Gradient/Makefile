CC = gcc
MPICC = mpicc

CFLAGS = -O3 -Wall -std=c99
MPI_CFLAGS = -O3 -Wall -std=c99

# Default target
all: serial parallel

# Serial executable
target: serial
serial: serial.c
	$(CC) $(CFLAGS) -o serial serial.c -lm

# Parallel executable
parallel: parallel.c
	$(MPICC) $(MPI_CFLAGS) -o parallel parallel.c

# Clean up build artifacts
.PHONY: clean all
clean:
	rm -f serial parallel *.o *.out *.txt
