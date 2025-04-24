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

# Strong scaling 1

mpirun -np 17 ./parallel input_2048.txt output_2048_16.txt
mpirun -np 33 ./parallel input_2048.txt output_2048_32.txt
mpirun -np 65 ./parallel input_2048.txt output_2048_64.txt
mpirun -np 129 ./parallel input_2048.txt output_2048_128.txt
mpirun -np 257 ./parallel input_2048.txt output_2048_256.txt

# Serial

./serial input_2048.txt output_2048_1.txt
./serial input_1024.txt output_1024_1.txt
./serial input_512.txt output_512_1.txt
./serial input_256.txt output_256_1.txt
./serial input_128.txt output_128_1.txt

# Strong scaling 2

mpirun -np 17 ./parallel input_1024.txt output_1024_16.txt
mpirun -np 33 ./parallel input_1024.txt output_1024_32.txt
mpirun -np 65 ./parallel input_1024.txt output_1024_64.txt
mpirun -np 129 ./parallel input_1024.txt output_1024_128.txt
mpirun -np 257 ./parallel input_1024.txt output_1024_256.txt

# Weak scaling 1

mpirun -np 17 ./parallel input_128.txt output_128_16.txt
mpirun -np 33 ./parallel input_256.txt output_256_32.txt
mpirun -np 65 ./parallel input_512.txt output_512_64.txt
mpirun -np 129 ./parallel input_1024.txt output_1024_128.txt
mpirun -np 257 ./parallel input_2048.txt output_2048_256.txt