// Görkem Kadir Solun 22003214
// MPI‐parallel CG for dense SPD; master (rank 0) only does I/O & final gather.
// P2P for p‐exchange before each SpMV, collective only for inner products.
// Follows Algorithm 3 of Selvitopi et al.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MASTER 0
#define DISTRIBUTE_TAG 100
#define DISTRIBUTE_TAG_A 101
#define DISTRIBUTE_TAG_B 102
#define P_SEND_TAG 200
#define RESULT_SEND_TAG 300

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2 || argc > 3)
    {
        if (world_rank == MASTER)
        {
            fprintf(stderr, "Usage: %s <input_file> [output_file]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const char *input_file_name = argv[1];
    const char *output_file_name = (argc == 3) ? argv[2] : "p_output.txt";

    if (world_size < 2)
    {
        if (world_rank == MASTER)
        {
            fprintf(stderr, "Need >=2 MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Split out a communicator for all the workers (ranks 1..size-1)
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank == MASTER ? MPI_UNDEFINED : 1,
                   world_rank,
                   &worker_comm);

    int p = world_size - 1;
    int sqrt_p = (int)sqrt(p);

    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int cart_rank, coords[2]; // α (row) and β (column)
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(cart_comm, coords[1], coords[0], &col_comm);

    int n;
    double *A = NULL, *b = NULL;

    if (world_rank == MASTER)
    {
        FILE *input_file = fopen(input_file_name, "r");
        if (!input_file)
        {
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(input_file, "%d", &n);

        if (n % sqrt_p != 0)
        {
            fprintf(stderr, "Error: n (%d) is not divisible by sqrt_p (%d)\n", n, sqrt_p);
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int w = 1; w < world_size; ++w)
        {
            MPI_Send(&n, 1, MPI_INT, w, DISTRIBUTE_TAG, MPI_COMM_WORLD);
        }

        A = malloc(n * n * sizeof(double));
        b = malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                fscanf(input_file, "%lf", &A[i * n + j]);
            }
        }
        for (int i = 0; i < n; ++i)
        {
            fscanf(input_file, "%lf", &b[i]);
        }
        fclose(input_file);

        int block_length = n / sqrt_p;
        int block_size = block_length * block_length;

        printf("block_length: %d, block_size: %d\n", block_length, block_size);

        for (int i = 0; i < sqrt_p; ++i)
        {
            for (int j = 0; j < sqrt_p; ++j)
            {
                // Send A_ij to the destination rank, A_αβ
                int destination_rank;
                MPI_Cart_rank(cart_comm, (int[]){i, j}, &destination_rank);
                double *temp_A = malloc(block_size * sizeof(double));
                for (int k = 0; k < block_length; ++k)
                {
                    memcpy(temp_A + k * block_length, A + (i * block_length + k) * n + (j * block_length), block_length * sizeof(double));
                }
                MPI_Send(temp_A, block_size, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_A, MPI_COMM_WORLD);

                // Send b_ji to the destination rank, b_βα
                MPI_Cart_rank(cart_comm, (int[]){j, i}, &destination_rank);
                MPI_Send(b + j * block_length + i, block_length / sqrt_p, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_B, MPI_COMM_WORLD);
            }
        }
        // P_ij knows A_ij and b_ji, P_αβ knows A_αβ and b_βα

        /* MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        double *result = malloc(n * sizeof(double));
        for (int w = 1; w < world_size; ++w)
        {
            int offset = (w - 1) * rows_per_worker;
            MPI_Recv(result + offset, rows_per_worker, MPI_DOUBLE, w, RESULT_SEND_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        FILE *output_file = fopen(output_file_name, "w");
        for (int i = 0; i < n; ++i)
        {
            fprintf(output_file, "%.8f\n", result[i]);
        }

        fprintf(output_file, "Elapsed time: %.6f s\n", t_end - t_start);

        fclose(output_file);

        free(A);
        free(b);
        free(result);*/
    }
    else
    {
        MPI_Recv(&n, 1, MPI_INT, MASTER, DISTRIBUTE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int block_length = n / sqrt_p;
        int block_size = block_length * block_length;

        double *A_ij = malloc(block_size * sizeof(double));
        MPI_Recv(A_ij, block_size, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *b_ji = malloc(block_length / sqrt_p * sizeof(double));
        MPI_Recv(b_ji, block_length / sqrt_p, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("A_ij: %p, b_ji: %p\n", A_ij, b_ji);

        MPI_Barrier(MPI_COMM_WORLD);

        printf("A_ij: %p, b_ji: %p\n", A_ij, b_ji);

        int worker_rank, worker_size;
        MPI_Comm_rank(worker_comm, &worker_rank);
        MPI_Comm_size(worker_comm, &worker_size);

        double *x_loc = calloc(block_length, sizeof(double));
        double *r_loc = malloc(block_length * sizeof(double));
        double *p_loc = malloc(block_length * sizeof(double));
        double *q_loc = malloc(block_length * sizeof(double));

        // Perform ring AABC to get b_j for all i by sending requests to same column peers
        double *b_j = malloc(block_length * sizeof(double));

        for (int i = 0; i < block_length; ++i)
        {
            b_j[i] = b_ji[i];
        }

        MPI_Request *requests = malloc(2 * (block_length - 1) * sizeof(MPI_Request));
        int request_counter = 0;

        printf("coords: %d %d\n", coords[0], coords[1]);

        int block_relative_coords[2];
        block_relative_coords[0] = coords[0] % block_length;
        block_relative_coords[1] = coords[1] % block_length;
        printf("block_relative_coords: %d %d\n", block_relative_coords[0], block_relative_coords[1]);

        for (int i = 0; i < block_length; ++i)
        {
            if (i == coords[1] % block_length)
            {
                continue;
            }
        }

        // initial r=b, p=r
        for (int i = 0; i < block_length; ++i)
        {
            r_loc[i] = b_j[i];
            p_loc[i] = r_loc[i];
        }

        // full p and r location
        double *p_full = malloc(n * sizeof(double));

        /* double r_dot, p_dot_q, r_dot_new, alpha, beta;
        for (int iteration = 0; iteration < n; ++iteration)
        {
            int my_offset = worker_rank * rows_per_worker;
            for (int i = 0; i < rows_per_worker; ++i)
            {
                p_full[my_offset + i] = p_loc[i];
            }

            // post receives from all peers, then sends
            MPI_Request *requests = malloc(2 * (worker_size - 1) * sizeof(MPI_Request));
            int request_counter = 0;
            for (int peer = 0; peer < worker_size; ++peer)
            {
                if (peer == worker_rank)
                {
                    continue;
                }

                int peer_offset = peer * rows_per_worker;
                MPI_Irecv(p_full + peer_offset, rows_per_worker, MPI_DOUBLE,
                          peer + 1, P_SEND_TAG, MPI_COMM_WORLD,
                          &requests[request_counter++]);
            }

            for (int peer = 0; peer < worker_size; ++peer)
            {
                if (peer == worker_rank)
                {
                    continue;
                }

                MPI_Isend(p_loc, rows_per_worker, MPI_DOUBLE,
                          peer + 1, P_SEND_TAG, MPI_COMM_WORLD,
                          &requests[request_counter++]);
            }

            if (request_counter > 0)
            {
                MPI_Waitall(request_counter, requests, MPI_STATUSES_IGNORE);
            }
            free(requests);

            // local SpMV q_loc = A_loc * p_full
            for (int i = 0; i < rows_per_worker; ++i)
            {
                double sum = 0;
                for (int j = 0; j < n; ++j)
                {
                    sum += A_loc[i * n + j] * p_full[j];
                }
                q_loc[i] = sum;
            }

            // local dot products
            p_dot_q = 0;
            r_dot = 0;
            for (int i = 0; i < rows_per_worker; ++i)
            {
                p_dot_q += p_loc[i] * q_loc[i];
                r_dot += r_loc[i] * r_loc[i];
            }

            double sum_pq, sum_rr;
            MPI_Allreduce(&p_dot_q, &sum_pq, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
            MPI_Allreduce(&r_dot, &sum_rr, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            if (iteration == 0)
            {
                alpha = sum_rr / sum_pq;
            }
            else
            {
                beta = sum_rr / r_dot;
                alpha = sum_rr / sum_pq;
            }

            // update
            for (int i = 0; i < rows_per_worker; ++i)
            {
                x_loc[i] += alpha * p_loc[i];
                r_loc[i] -= alpha * q_loc[i];
            }

            r_dot_new = 0;
            for (int i = 0; i < rows_per_worker; ++i)
            {
                r_dot_new += r_loc[i] * r_loc[i];
            }
            MPI_Allreduce(MPI_IN_PLACE, &r_dot_new, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            if (sqrt(r_dot_new) < 1e-16)
            {
                break;
            }

            beta = r_dot_new / sum_rr;
            for (int i = 0; i < rows_per_worker; ++i)
            {
                p_loc[i] = r_loc[i] + beta * p_loc[i];
            }
            r_dot = r_dot_new;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Send(x_loc, block_length, MPI_DOUBLE,
                 MASTER, RESULT_SEND_TAG, MPI_COMM_WORLD);

        free(A_ij);
        free(b_ji);
        free(x_loc);
        free(r_loc);
        free(p_loc);
        free(q_loc);
        free(p_full);
        MPI_Comm_free(&worker_comm);
        */
    }

    MPI_Finalize();
    return 0;
}
