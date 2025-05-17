#!/bin/bash

#SBATCH -J "Example_Run_Egitim187" # Name of the job (*)
#SBATCH -C weka # High-performance option for ORFOZ queue (*) – please don’t change this
#SBATCH -A egitim187 # Account name (*)
#SBATCH -p orfoz # Partition/Queue name (*)
#SBATCH -o ./output.out # Output file (all console outputs go here, e.g., prints, flushes)
#SBATCH -e ./error.err # Error file (if any error occurs, it goes here)
#SBATCH --open-mode=append  # File open mode (append: concatenates new output to the existing file)
#SBATCH -n 330 # Number of cores you request (*)
#SBATCH -N 3 # Number of nodes you request (*)
#SBATCH -t 0-00:30:00 # Estimated runtime (DAY-HOUR:MINUTE:SECOND)
#SBATCH --mail-type=END,FAIL # Email notifications when the job finishes or fails
#SBATCH --mail-user=kadir.solun@ug.bilkent.edu.tr # Email address for notifications

module load oneapi
ulimit -c unlimited  # Set ulimit to unlimited to avoid any stack limit errors

make

# You can also use srun, but in the ORFOZ queue you can request predefined cores per node (e.g., 55 or 110).
# If you want to use a different number of cores (like 64, 128, etc.), specify it via mpirun -np.

# sample input files
mpirun -np 5 ./parallel input_8_m.txt input_8_v.txt output_8_5_2d.txt
mpirun -np 5 ./row_parallel input_8_m.txt input_8_v.txt output_8_5_1d.txt


# Strong scaling 1

mpirun -np 5 ./parallel input_4096_m.txt input_4096_v.txt output_4096_5_2d_strong.txt
mpirun -np 5 ./row_parallel input_4096_m.txt input_4096_v.txt output_4096_5_1d_strong.txt

mpirun -np 17 ./parallel input_4096_m.txt input_4096_v.txt output_4096_16_2d_strong.txt
mpirun -np 17 ./row_parallel input_4096_m.txt input_4096_v.txt output_4096_16_1d_strong.txt

mpirun -np 65 ./parallel input_4096_m.txt input_4096_v.txt output_4096_64_2d_strong.txt
mpirun -np 65 ./row_parallel input_4096_m.txt input_4096_v.txt output_4096_64_1d_strong.txt

mpirun -np 257 ./parallel input_4096_m.txt input_4096_v.txt output_4096_256_2d_strong.txt
mpirun -np 257 ./row_parallel input_4096_m.txt input_4096_v.txt output_4096_256_1d_strong.txt

# Strong scaling 2

mpirun -np 5 ./parallel input_8192_m.txt input_8192_v.txt output_8192_5_2d_strong.txt
mpirun -np 5 ./row_parallel input_8192_m.txt input_8192_v.txt output_8192_5_1d_strong.txt

mpirun -np 17 ./parallel input_8192_m.txt input_8192_v.txt output_8192_16_2d_strong.txt
mpirun -np 17 ./row_parallel input_8192_m.txt input_8192_v.txt output_8192_16_1d_strong.txt

mpirun -np 65 ./parallel input_8192_m.txt input_8192_v.txt output_8192_64_2d_strong.txt
mpirun -np 65 ./row_parallel input_8192_m.txt input_8192_v.txt output_8192_64_1d_strong.txt

mpirun -np 257 ./parallel input_8192_m.txt input_8192_v.txt output_8192_256_2d_strong.txt
mpirun -np 257 ./row_parallel input_8192_m.txt input_8192_v.txt output_8192_256_1d_strong.txt

# Weak scaling 1

mpirun -np 5 ./parallel input_1024_m.txt input_1024_v.txt output_1024_4_2d_weak.txt
mpirun -np 5 ./row_parallel input_1024_m.txt input_1024_v.txt output_1024_4_1d_weak.txt

mpirun -np 17 ./parallel input_2048_m.txt input_2048_v.txt output_2048_16_2d_weak.txt
mpirun -np 17 ./row_parallel input_2048_m.txt input_2048_v.txt output_2048_16_1d_weak.txt

mpirun -np 65 ./parallel input_4096_m.txt input_4096_v.txt output_4096_64_2d_weak.txt
mpirun -np 65 ./row_parallel input_4096_m.txt input_4096_v.txt output_4096_64_1d_weak.txt

mpirun -np 257 ./parallel input_8192_m.txt input_8192_v.txt output_8192_256_2d_weak.txt
mpirun -np 257 ./row_parallel input_8192_m.txt input_8192_v.txt output_8192_256_1d_weak.txt