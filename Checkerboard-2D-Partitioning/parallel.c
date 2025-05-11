// Görkem Kadir Solun 22003214
// MPI‐parallel CG for dense SPD; master (rank 0) only does I/O & final gather.
// P2P for p‐exchange before each SpMV, collective only for inner products.
// Follows Algorithm 3 of Selvitopi et al.

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
#define P_RECV_TAG 201
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

    int p = world_size - 1;
    int sqrt_p = (int)sqrt(p);

    // Create worker communicator first
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank == MASTER ? MPI_UNDEFINED : 1,
                   world_rank,
                   &worker_comm);

    // Only create Cartesian communicator for worker processes
    MPI_Comm cart_comm = MPI_COMM_NULL, row_comm = MPI_COMM_NULL, col_comm = MPI_COMM_NULL;

    int cart_rank = -1, coords[2] = {-1, -1};

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
                // Calculate destination rank in worker communicator
                int destination_rank = i * sqrt_p + j + 1; // +1 because rank 0 is master

                // Send A_ij to the destination rank
                double *temp_A = malloc(block_size * sizeof(double));
                for (int k = 0; k < block_length; ++k)
                {
                    memcpy(temp_A + k * block_length, A + (i * block_length + k) * n + (j * block_length), block_length * sizeof(double));
                }
                MPI_Send(temp_A, block_size, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_A, MPI_COMM_WORLD);
                free(temp_A);

                // Send b_ji to the destination rank
                MPI_Send(b + j * block_length + i, block_length / sqrt_p, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_B, MPI_COMM_WORLD);
            }
        }
        // P_ij knows A_ij and b_ji, P_αβ knows A_αβ and b_βα

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        double *result = malloc(n * sizeof(double));
        for (int i = 0; i < sqrt_p; ++i)
        {
            for (int j = 0; j < sqrt_p; ++j)
            {
                MPI_Recv(result + i * block_length + j, block_length, MPI_DOUBLE, i * sqrt_p + j + 1, RESULT_SEND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
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
        free(result);
    }
    else
    {
        printf("Worker %d: Starting initialization\n", world_rank);

        int dims[2] = {sqrt_p, sqrt_p};
        int periods[2] = {0, 0};
        printf("Worker %d: About to create cart_comm\n", world_rank);
        MPI_Cart_create(worker_comm, 2, dims, periods, 1, &cart_comm);
        printf("Worker %d: Created cart_comm\n", world_rank);

        MPI_Comm_rank(cart_comm, &cart_rank);
        MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
        printf("Worker %d: Got cart_rank=%d, coords=[%d,%d]\n", world_rank, cart_rank, coords[0], coords[1]);

        MPI_Comm_split(cart_comm, coords[0], coords[1], &row_comm);
        MPI_Comm_split(cart_comm, coords[1], coords[0], &col_comm);
        printf("Worker %d: Created row and col comms\n", world_rank);

        MPI_Recv(&n, 1, MPI_INT, MASTER, DISTRIBUTE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Worker %d: Received n=%d\n", world_rank, n);

        int block_length = n / sqrt_p;
        int block_size = block_length * block_length;
        printf("Worker %d: block_length=%d, block_size=%d\n", world_rank, block_length, block_size);

        double *A_ij = malloc(block_size * sizeof(double));
        printf("Worker %d: Allocated A_ij\n", world_rank);
        MPI_Recv(A_ij, block_size, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Worker %d: Received A_ij\n", world_rank);

        double *b_ji = malloc(block_length / sqrt_p * sizeof(double));
        printf("Worker %d: Allocated b_ji\n", world_rank);
        MPI_Recv(b_ji, block_length / sqrt_p, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Worker %d: Received b_ji\n", world_rank);

        for (int i = 0; i < block_length; ++i)
        {
            for (int j = 0; j < block_length; ++j)
            {
                printf("Worker %d: A_ij[%d][%d] = %f\n", world_rank, i, j, A_ij[i * block_length + j]);
            }
        }
        for (int i = 0; i < block_length / sqrt_p; ++i)
        {
            printf("Worker %d: b_ji[%d] = %f\n", world_rank, i, b_ji[i]);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        printf("Worker %d: coords: %d %d\n", world_rank, coords[0], coords[1]);

        int worker_rank, worker_size;
        MPI_Comm_rank(worker_comm, &worker_rank);
        MPI_Comm_size(worker_comm, &worker_size);

        double *x_loc = calloc(block_length, sizeof(double));
        double *r_loc = malloc(block_length * sizeof(double));
        double *p_loc = malloc(block_length * sizeof(double));
        double *q_loc = malloc(block_length * sizeof(double));

        double *b_j = malloc(block_length * sizeof(double));

        // Perform ring AABC to get b_j for all i by sending requests to same column peers
        MPI_Request *requests = malloc(2 * (sqrt_p - 1) * sizeof(MPI_Request));
        int request_counter = 0;

        // First post all receives
        for (int i = 0; i < sqrt_p; ++i)
        {
            if (i == coords[0])
            {
                continue;
            }

            int source_rank = i * sqrt_p + coords[1] + 1;
            MPI_Irecv(b_j + i * block_length / sqrt_p, block_length / sqrt_p, MPI_DOUBLE,
                      source_rank, P_SEND_TAG, MPI_COMM_WORLD, &requests[request_counter++]);
            printf("Worker %d: Posted receive from worker %d\n", world_rank, source_rank);
        }
        // Then post all sends
        for (int i = 0; i < sqrt_p; ++i)
        {
            if (i == coords[0])
            {
                continue;
            }

            int destination_rank = i * sqrt_p + coords[1] + 1;
            MPI_Isend(b_ji, block_length / sqrt_p, MPI_DOUBLE,
                      destination_rank, P_SEND_TAG, MPI_COMM_WORLD, &requests[request_counter++]);
            printf("Worker %d: Posted send to worker %d\n", world_rank, destination_rank);
        }

        if (request_counter > 0)
        {
            printf("Worker %d: Waiting for %d communications to complete\n", world_rank, request_counter);
            MPI_Waitall(request_counter, requests, MPI_STATUSES_IGNORE);
        }
        free(requests);
        printf("Worker %d: All communications completed\n", world_rank);

        // Copy local b_ji to b_j
        memcpy(b_j + coords[0] * block_length / sqrt_p, b_ji, block_length / sqrt_p * sizeof(double));

        // Print b_j
        for (int i = 0; i < block_length; ++i)
        {
            printf("Worker %d: b_j[%d] = %f\n", world_rank, i, b_j[i]);
        }

        // initial r=b, p=r
        for (int i = 0; i < block_length; ++i)
        {
            r_loc[i] = b_j[i];
            p_loc[i] = r_loc[i];
        }

        double r_dot, p_dot_q, r_dot_new, alpha, beta;
        for (int iteration = 0; iteration < n; ++iteration)
        {   

            // local SpMV q_loc = A_loc * p_loc
            for (int i = 0; i < block_length; ++i)
            {
                double sum = 0;
                for (int j = 0; j < block_length; ++j)
                {
                    sum += A_ij[i * block_length + j] * p_loc[j];
                }
                q_loc[i] = sum;
            }

            // print q_loc
            for (int i = 0; i < block_length; ++i)
            {
                printf("Iteration %d: Worker %d: q_loc[%d] = %f\n", iteration, world_rank, i, q_loc[i]);
            }

            // local dot products
            p_dot_q = 0;
            r_dot = 0;
            for (int i = 0; i < block_length; ++i)
            {
                p_dot_q += p_loc[i] * q_loc[i];
                r_dot += r_loc[i] * r_loc[i];
            }

            printf("Iteration %d: Worker %d: local p_dot_q: %f, local r_dot: %f\n", iteration, world_rank, p_dot_q, r_dot);

            double sum_pq, sum_rr;
            MPI_Allreduce(&p_dot_q, &sum_pq, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
            MPI_Allreduce(&r_dot, &sum_rr, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            printf("Iteration %d: Worker %d: sum_pq: %f, sum_rr: %f\n", iteration, world_rank, sum_pq, sum_rr);

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
            for (int i = 0; i < block_length; ++i)
            {
                x_loc[i] += alpha * p_loc[i];
                r_loc[i] -= alpha * q_loc[i];
            }

            r_dot_new = 0;
            for (int i = 0; i < block_length; ++i)
            {
                r_dot_new += r_loc[i] * r_loc[i];
            }
            MPI_Allreduce(MPI_IN_PLACE, &r_dot_new, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            printf("Iteration %d: Worker %d: r_dot_new: %f\n", iteration, world_rank, r_dot_new);

            if (sqrt(r_dot_new) < 1e-8)
            {
                break;
            }

            beta = r_dot_new / sum_rr;
            for (int i = 0; i < block_length; ++i)
            {
                p_loc[i] = r_loc[i] + beta * p_loc[i];
            }
            r_dot = r_dot_new;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Beware the location of x_loc
        MPI_Send(x_loc, block_length, MPI_DOUBLE,
                 MASTER, RESULT_SEND_TAG, MPI_COMM_WORLD);

        free(A_ij);
        free(b_ji);
        free(x_loc);
        free(r_loc);
        free(p_loc);
        free(q_loc);
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();
    return 0;
}
