CC = gcc
MPICC = mpicc

CFLAGS = -Wall -O2
MPICFLAGS = -Wall -O2

all: average-serial average-mpi-ppv1 average-mpi-ppv2 bucket_average-serial bucket_average-mpi

average-serial: average-serial.c
	$(CC) $(CFLAGS) -o $@ $<

average-mpi-ppv1: average-mpi-ppv1.c
	$(MPICC) $(MPICFLAGS) -o $@ $<

average-mpi-ppv2: average-mpi-ppv2.c
	$(MPICC) $(MPICFLAGS) -o $@ $<

bucket_average-serial: bucket_average-serial.c
	$(CC) $(CFLAGS) -o $@ $<

bucket_average-mpi: bucket_average-mpi.c
	$(MPICC) $(MPICFLAGS) -o $@ $<

clean:
	rm -f average-serial average-mpi-ppv1 average-mpi-ppv2 bucket_average-serial bucket_average-mpi *.o

.PHONY: all clean
