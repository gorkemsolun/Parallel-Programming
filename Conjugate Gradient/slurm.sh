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

mpirun -np 16 ./parallel input_2048.txt
mpirun -np 32 ./parallel input_2048.txt
mpirun -np 64 ./parallel input_2048.txt
mpirun -np 128 ./parallel input_2048.txt
mpirun -np 256 ./parallel input_2048.txt

# Serial

./serial input_2048.txt
./serial input_1024.txt
./serial input_512.txt
./serial input_256.txt
./serial input_128.txt

# Strong scaling 2

mpirun -np 16 ./parallel input_1024.txt
mpirun -np 32 ./parallel input_1024.txt
mpirun -np 64 ./parallel input_1024.txt
mpirun -np 128 ./parallel input_1024.txt
mpirun -np 256 ./parallel input_1024.txt

# Weak scaling 1

mpirun -np 16 ./parallel input_128.txt
mpirun -np 32 ./parallel input_256.txt
mpirun -np 64 ./parallel input_512.txt
mpirun -np 128 ./parallel input_1024.txt
mpirun -np 256 ./parallel input_2048.txt