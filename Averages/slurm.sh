#!/bin/bash

#SBATCH -J "Example_Run_Egitim187" # Name of the job (*)
#SBATCH -C weka # High-performance option for ORFOZ queue (*) – please don’t change this
#SBATCH -A egitim187 # Account name (*)
#SBATCH -p orfoz # Partition/Queue name (*)
#SBATCH -o ./output.out # Output file (all console outputs go here, e.g., prints, flushes)
#SBATCH -e ./error.err # Error file (if any error occurs, it goes here)
#SBATCH --open-mode=append  # File open mode (append: concatenates new output to the existing file)
#SBATCH -n 55 # Number of cores you request (*)
#SBATCH -N 1 # Number of nodes you request (*)
#SBATCH -t 0-00:30:00 # Estimated runtime (DAY-HOUR:MINUTE:SECOND)
#SBATCH --mail-type=END,FAIL # Email notifications when the job finishes or fails
#SBATCH --mail-user=kadir.solun@ug.bilkent.edu.tr # Email address for notifications

module load oneapi
ulimit -c unlimited  # Set ulimit to unlimited to avoid any stack limit errors

make

# You can also use srun, but in the ORFOZ queue you can request predefined cores per node (e.g., 55 or 110).
# If you want to use a different number of cores (like 64, 128, etc.), specify it via mpirun -np.

# mpirun -np 64 ./example

mpirun -np 1 ./average-serial input.txt output_serial.txt
mpirun -np 20 ./average-mpi-ppv1 input.txt output_ppv1.txt
mpirun -np 20 ./average-mpi-ppv2 input.txt output_ppv2.txt
mpirun -np 20 ./bucket_average-mpi input_buckets.txt output_buckets.txt
mpirun -np 1 ./bucket_average-serial input_buckets.txt output_buckets_serial.txt