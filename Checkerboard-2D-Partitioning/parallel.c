// 2D Checkerboard Parallel Matrix-Vector Multiplication (SpMV)
// Master: rank 0 handles I/O & scatter/gather; Workers: ranks 1..P-1 form a √p×√p mesh

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASTER 0
#define DIST_TAG    100
#define TAG_EXPAND  101
#define TAG_FOLD    102
#define TAG_TRANS   103
#define TAG_RESULT  104

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        if (world_rank == MASTER)
            fprintf(stderr, "Need at least 2 MPI processes.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (argc < 2) {
        if (world_rank == MASTER)
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int orig_rank = world_rank;
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank == MASTER ? MPI_UNDEFINED : 1,
                   world_rank,
                   &worker_comm);

    int n;
    int dims[2];
    int block_rows, block_cols;
    double *A = NULL, *X = NULL, *Y = NULL;

    if (world_rank == MASTER) {
        // Master: read input matrix A and vector X
        FILE *fp = fopen(argv[1], "r");
        if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fscanf(fp, "%d", &n);
        A = malloc(n * n * sizeof(double));
        X = malloc(n * sizeof(double));
        for (int i = 0; i < n*n; i++) fscanf(fp, "%lf", &A[i]);
        for (int i = 0; i < n;   i++) fscanf(fp, "%lf", &X[i]);
        fclose(fp);

        // Determine the 2D grid dimensions
        int num_workers = world_size - 1;
        MPI_Dims_create(num_workers, 2, dims);  // dims[0]×dims[1] = num_workers
        block_rows = n / dims[0];
        block_cols = n / dims[1];
        if (n % dims[0] != 0 || n % dims[1] != 0) {
            fprintf(stderr, "Error: matrix dimension not divisible by grid dims\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Scatter A-blocks and X-slices to each worker
        for (int wr = 1; wr < world_size; wr++) {
            int idx = wr - 1;
            int alpha = idx / dims[1];
            int beta  = idx % dims[1];

            MPI_Send(&n,     1, MPI_INT,    wr, DIST_TAG, MPI_COMM_WORLD);
            MPI_Send(dims,   2, MPI_INT,    wr, DIST_TAG, MPI_COMM_WORLD);

            // pack A-block
            double *Ablk = malloc(block_rows * block_cols * sizeof(double));
            for (int i = 0; i < block_rows; i++) {
                for (int j = 0; j < block_cols; j++) {
                    Ablk[i*block_cols + j]
                        = A[(alpha*block_rows + i)*n + (beta*block_cols + j)];
                }
            }
            MPI_Send(Ablk, block_rows*block_cols, MPI_DOUBLE,
                     wr, DIST_TAG, MPI_COMM_WORLD);
            free(Ablk);

            // send corresponding X-slice
            MPI_Send(X + beta*block_cols,
                     block_cols, MPI_DOUBLE,
                     wr, DIST_TAG, MPI_COMM_WORLD);
        }
        free(A);
        free(X);

        // Gather Y-blocks from row-0 grid processes and assemble Y
        Y = malloc(n * sizeof(double));
        for (int beta = 0; beta < dims[1]; beta++) {
            MPI_Recv(Y + beta*block_rows,
                     block_rows, MPI_DOUBLE,
                     1 + beta, TAG_RESULT,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Output the result
        for (int i = 0; i < n; i++)
            printf("%f\n", Y[i]);
        free(Y);

    } else {
        // Worker: form 2D Cartesian grid
        int wsize;
        MPI_Comm_size(worker_comm, &wsize);
        MPI_Dims_create(wsize, 2, dims);
        MPI_Comm grid_comm;
        int periods[2] = {0, 0}, reorder = 1;
        MPI_Cart_create(worker_comm, 2, dims, periods, reorder, &grid_comm);

        int grid_rank;
        MPI_Comm_rank(grid_comm, &grid_rank);
        int coord[2];
        MPI_Cart_coords(grid_comm, grid_rank, 2, coord);

        // Receive metadata and local blocks
        MPI_Recv(&n,     1, MPI_INT, MASTER, DIST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(dims,   2, MPI_INT, MASTER, DIST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        block_rows = n / dims[0];
        block_cols = n / dims[1];

        double *A_loc = malloc(block_rows * block_cols * sizeof(double));
        double *X_loc = malloc(block_cols * sizeof(double));
        MPI_Recv(A_loc, block_rows*block_cols, MPI_DOUBLE,
                 MASTER, DIST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(X_loc, block_cols, MPI_DOUBLE,
                 MASTER, DIST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 1) EXPAND: propagate X_loc down each column
        if (coord[0] == 0) {
            for (int i = 1; i < dims[0]; i++) {
                int dst_coords[2] = {i, coord[1]};
                int dst_rank;
                MPI_Cart_rank(grid_comm, dst_coords, &dst_rank);
                MPI_Send(X_loc, block_cols, MPI_DOUBLE,
                         dst_rank, TAG_EXPAND, grid_comm);
            }
        } else {
            MPI_Recv(X_loc, block_cols, MPI_DOUBLE,
                     MPI_ANY_SOURCE, TAG_EXPAND,
                     grid_comm, MPI_STATUS_IGNORE);
        }

        // 2) LOCAL multiply: Y_loc = A_loc × X_loc
        double *Y_loc = malloc(block_rows * sizeof(double));
        for (int i = 0; i < block_rows; i++) {
            double sum = 0;
            for (int j = 0; j < block_cols; j++)
                sum += A_loc[i*block_cols + j] * X_loc[j];
            Y_loc[i] = sum;
        }

        // 3) FOLD: sum Y_loc across columns to aggregator at beta=0
        double *Y_sum = NULL;
        if (coord[1] == 0) {
            Y_sum = malloc(block_rows * sizeof(double));
            memcpy(Y_sum, Y_loc, block_rows * sizeof(double));
            double *tmp = malloc(block_rows * sizeof(double));
            for (int j = 1; j < dims[1]; j++) {
                int src_coords[2] = {coord[0], j};
                int src_rank;
                MPI_Cart_rank(grid_comm, src_coords, &src_rank);
                MPI_Recv(tmp, block_rows, MPI_DOUBLE,
                         src_rank, TAG_FOLD, grid_comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < block_rows; i++)
                    Y_sum[i] += tmp[i];
            }
            free(tmp);
        } else {
            int agg_coords[2] = {coord[0], 0};
            int agg_rank;
            MPI_Cart_rank(grid_comm, agg_coords, &agg_rank);
            MPI_Send(Y_loc, block_rows, MPI_DOUBLE,
                     agg_rank, TAG_FOLD, grid_comm);
        }

        // 4) TRANSPOSE: row-aggregators send their sums to row 0
        if (coord[1] == 0) {
            int dest_coords[2] = {0, coord[0]};
            int dest_rank;
            MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
            MPI_Send(Y_sum, block_rows, MPI_DOUBLE,
                     dest_rank, TAG_TRANS, grid_comm);
        }

        // Receive final Y slice on row=0 and send to master
        if (coord[0] == 0) {
            double *Y_final = malloc(block_rows * sizeof(double));
            MPI_Recv(Y_final, block_rows, MPI_DOUBLE,
                     MPI_ANY_SOURCE, TAG_TRANS,
                     grid_comm, MPI_STATUS_IGNORE);
            MPI_Send(Y_final, block_rows, MPI_DOUBLE,
                     MASTER, TAG_RESULT, MPI_COMM_WORLD);
            free(Y_final);
        }

        // Cleanup
        free(Y_loc);
        if (coord[1] == 0) free(Y_sum);
        free(A_loc);
        free(X_loc);
        MPI_Comm_free(&grid_comm);
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
