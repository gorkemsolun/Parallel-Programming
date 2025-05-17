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
#define DISTRIBUTE_TAG_n 100
#define DISTRIBUTE_TAG_A 101
#define DISTRIBUTE_TAG_B 102
#define P_DISTRIBUTE_TAG 200
#define Q_DISTRIBUTE_TAG 201
#define Q_TRANSPOSE_TAG 202
#define RESULT_TAG 300

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

    int worker_size = world_size - 1;
    int worker_size_sqrt = (int)sqrt(worker_size);
    int worker_rank;

    // Create worker communicator
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
                   world_rank == MASTER ? MPI_UNDEFINED : 1,
                   world_rank,
                   &worker_comm);

    if (world_rank != MASTER)
    {
        MPI_Comm_rank(worker_comm, &worker_rank);
        MPI_Comm_size(worker_comm, &worker_size);
    }

    int n;
    double *A = NULL, *b = NULL;

    int block_length, block_length_separated, block_size, destination_rank;

    if (world_rank == MASTER)
    {
        FILE *input_file = fopen(input_file_name, "r");
        if (!input_file)
        {
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(input_file, "%d", &n);

        if (n % worker_size_sqrt != 0)
        {
            fprintf(stderr, "Error: n (%d) is not divisible by worker_size_sqrt (%d)\n", n, worker_size_sqrt);
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (n % worker_size != 0)
        {
            fprintf(stderr, "Error: n (%d) is not divisible by worker_size (%d)\n", n, worker_size);
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int w = 1; w < world_size; ++w)
        {
            MPI_Send(&n, 1, MPI_INT, w, DISTRIBUTE_TAG_n, MPI_COMM_WORLD);
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

        block_length = n / worker_size_sqrt;
        block_length_separated = n / worker_size;
        block_size = block_length * block_length;

        for (int i = 0; i < worker_size_sqrt; ++i)
        {
            for (int j = 0; j < worker_size_sqrt; ++j)
            {
                destination_rank = i * worker_size_sqrt + j + 1; // +1 because rank 0 is master

                // Send A_ij to the destination rank
                double *temp_A = malloc(block_size * sizeof(double));
                for (int k = 0; k < block_length; ++k)
                {
                    memcpy(temp_A + k * block_length, A + (i * block_length + k) * n + (j * block_length), block_length * sizeof(double));
                }
                MPI_Send(temp_A, block_size, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_A, MPI_COMM_WORLD);
                free(temp_A);

                // Send b_ji to the destination rank
                MPI_Send(b + j * block_length + i * block_length_separated, block_length_separated, MPI_DOUBLE, destination_rank, DISTRIBUTE_TAG_B, MPI_COMM_WORLD);
            }
        }
        // P_ij knows A_ij and b_ji, P_αβ knows A_αβ and b_βα

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();

        double *result = malloc(n * sizeof(double));
        for (int i = 0; i < worker_size_sqrt; ++i)
        {
            for (int j = 0; j < worker_size_sqrt; ++j)
            {
                MPI_Recv(result + i * block_length_separated + j * block_length, block_length_separated, MPI_DOUBLE, i * worker_size_sqrt + j + 1, RESULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        MPI_Recv(&n, 1, MPI_INT, MASTER, DISTRIBUTE_TAG_n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* printf("worker %d n: %d\n", worker_rank, n); */

        block_length = n / worker_size_sqrt;
        block_length_separated = n / worker_size;
        block_size = block_length * block_length;

        double *A_ij = malloc(block_size * sizeof(double));
        MPI_Recv(A_ij, block_size, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double *b_ji = malloc(block_length_separated * sizeof(double));
        MPI_Recv(b_ji, block_length_separated, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* printf("worker %d A_ij[0]: %f\n", worker_rank, A_ij[0]);
        printf("worker %d A_ij[1]: %f\n", worker_rank, A_ij[1]);
        printf("worker %d b_ji[0]: %f\n", worker_rank, b_ji[0]);
        printf("worker %d b_ji[1]: %f\n", worker_rank, b_ji[1]); */

        MPI_Barrier(MPI_COMM_WORLD);

        double *x_local = calloc(block_length_separated, sizeof(double));
        double *r_local = malloc(block_length_separated * sizeof(double));
        double *p_local = malloc(block_length_separated * sizeof(double));
        double *q_local = malloc(block_length_separated * sizeof(double));
        double *q_local_transposed = malloc(block_length_separated * sizeof(double));
        double *p_block = malloc(block_length * sizeof(double));
        double *q_block = malloc(block_length * sizeof(double));
        double *q_block_folded = malloc(block_length * sizeof(double));

        for (int i = 0; i < block_length_separated; ++i)
        {
            r_local[i] = b_ji[i];
            p_local[i] = r_local[i];
        }

        free(b_ji);

        double alpha, beta;
        double r_dot_r, r_dot_r_local = 0;
        double p_dot_q, q_dot_q, p_dot_q_local, q_dot_q_local;
        // double *p_dot_q_q_dot_q = malloc(2 * sizeof(double)), *p_dot_q_q_dot_q_local = malloc(2 * sizeof(double));

        for (int i = 0; i < block_length_separated; ++i)
        {
            r_dot_r_local += r_local[i] * r_local[i];
        }
        MPI_Allreduce(&r_dot_r_local, &r_dot_r, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

        while (r_dot_r > 1e-12)
        {
            int worker_column_index = worker_rank % worker_size_sqrt;
            int worker_row_index = worker_rank / worker_size_sqrt;

            MPI_Request requests[2 * (worker_size_sqrt - 1)];
            int request_index = 0;

            for (int i = 0; i < worker_size_sqrt; ++i)
            {
                destination_rank = i * worker_size_sqrt + worker_column_index;
                int block_offset = i * block_length_separated;

                if (i == worker_row_index)
                {
                    memcpy(p_block + block_offset,
                           p_local,
                           block_length_separated * sizeof(double));
                }
                else
                {
                    MPI_Irecv(p_block + block_offset,
                              block_length_separated, MPI_DOUBLE,
                              destination_rank, P_DISTRIBUTE_TAG,
                              worker_comm,
                              &requests[request_index++]);

                    MPI_Isend(p_local,
                              block_length_separated, MPI_DOUBLE,
                              destination_rank, P_DISTRIBUTE_TAG,
                              worker_comm,
                              &requests[request_index++]);
                }
            }

            MPI_Waitall(request_index, requests, MPI_STATUSES_IGNORE);

            /* printf("worker %d p_block[0]: %f\n", worker_rank, p_block[0]);
            printf("worker %d p_block[1]: %f\n", worker_rank, p_block[1]);
            printf("worker %d p_block[2]: %f\n", worker_rank, p_block[2]);
            printf("worker %d p_block[3]: %f\n", worker_rank, p_block[3]); */

            for (int i = 0; i < block_length; ++i)
            {
                double sum = 0;
                for (int j = 0; j < block_length; ++j)
                {
                    sum += A_ij[i * block_length + j] * p_block[j];
                }
                q_block[i] = sum;
            }

            /* printf("worker %d q_block[0]: %f\n", worker_rank, q_block[0]);
            printf("worker %d q_block[1]: %f\n", worker_rank, q_block[1]);
            printf("worker %d q_block[2]: %f\n", worker_rank, q_block[2]);
            printf("worker %d q_block[3]: %f\n", worker_rank, q_block[3]); */

            memcpy(q_block_folded, q_block, block_length * sizeof(double));
            int worker_row_start_rank = worker_row_index * worker_size_sqrt;

            int peer_count = worker_size_sqrt - 1;
            MPI_Request *reqs = malloc(2 * peer_count * sizeof(MPI_Request));
            double **temp_buffers = malloc(peer_count * sizeof(double *));

            request_index = 0;
            int buffer_index = 0;

            for (int j = 0; j < worker_size_sqrt; ++j)
            {
                if (j == worker_column_index)
                {
                    continue;
                }

                destination_rank = worker_row_start_rank + j;

                double *temp_q_block = malloc(block_length * sizeof(double));
                temp_buffers[buffer_index++] = temp_q_block;

                MPI_Irecv(temp_q_block, block_length, MPI_DOUBLE,
                          destination_rank, Q_DISTRIBUTE_TAG,
                          worker_comm, &reqs[request_index++]);

                MPI_Isend(q_block, block_length, MPI_DOUBLE,
                          destination_rank, Q_DISTRIBUTE_TAG,
                          worker_comm, &reqs[request_index++]);
            }

            MPI_Waitall(2 * peer_count, reqs, MPI_STATUSES_IGNORE);

            for (int k = 0; k < peer_count; ++k)
            {
                double *temp_q_block = temp_buffers[k];
                for (int i = 0; i < block_length; ++i)
                {
                    q_block_folded[i] += temp_q_block[i];
                }
                free(temp_q_block);
            }

            free(temp_buffers);
            free(reqs);

            /* printf("worker %d q_block_folded[0]: %f\n", worker_rank, q_block_folded[0]);
            printf("worker %d q_block_folded[1]: %f\n", worker_rank, q_block_folded[1]);
            printf("worker %d q_block_folded[2]: %f\n", worker_rank, q_block_folded[2]);
            printf("worker %d q_block_folded[3]: %f\n", worker_rank, q_block_folded[3]); */

            // q_local = q_block_folded[worker_column_index]
            memcpy(q_local, q_block_folded + worker_column_index * block_length_separated, block_length_separated * sizeof(double));

            /* printf("worker %d q_local[0]: %f\n", worker_rank, q_local[0]);
            printf("worker %d q_local[1]: %f\n", worker_rank, q_local[1]); */

            // Transpose q_locals
            destination_rank = worker_column_index * worker_size_sqrt + worker_row_index;
            MPI_Sendrecv(q_local, block_length_separated, MPI_DOUBLE, destination_rank, Q_TRANSPOSE_TAG, q_local_transposed, block_length_separated, MPI_DOUBLE, destination_rank, Q_TRANSPOSE_TAG, worker_comm, MPI_STATUS_IGNORE);

            /* printf("worker %d q_local_transposed[0]: %f\n", worker_rank, q_local_transposed[0]);
            printf("worker %d q_local_transposed[1]: %f\n", worker_rank, q_local_transposed[1]); */

            p_dot_q_local = 0;
            for (int i = 0; i < block_length_separated; ++i)
            {
                p_dot_q_local += p_local[i] * q_local_transposed[i];
            }
            MPI_Allreduce(&p_dot_q_local, &p_dot_q, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            alpha = r_dot_r / p_dot_q;

            for (int i = 0; i < block_length_separated; ++i)
            {
                r_local[i] -= alpha * q_local_transposed[i];
                x_local[i] += alpha * p_local[i];
            }

            r_dot_r_local = 0;
            double new_r_dot_r = 0;
            for (int i = 0; i < block_length_separated; ++i)
            {
                r_dot_r_local += r_local[i] * r_local[i];
            }
            MPI_Allreduce(&r_dot_r_local, &new_r_dot_r, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            beta = new_r_dot_r / r_dot_r;
            r_dot_r = new_r_dot_r;
            for (int i = 0; i < block_length_separated; ++i)
            {
                p_local[i] = r_local[i] + beta * p_local[i];
            }

            /* printf("worker %d r_dot_r: %f\n", worker_rank, r_dot_r); */

            /* p_dot_q_q_dot_q_local[0] = 0;
            p_dot_q_q_dot_q_local[1] = 0;
            for (int i = 0; i < block_length_separated; ++i)
            {
                p_dot_q_q_dot_q_local[0] += p_local[i] * q_local[i];
                p_dot_q_q_dot_q_local[1] += q_local[i] * q_local[i];
            }
            MPI_Allreduce(p_dot_q_q_dot_q_local, p_dot_q_q_dot_q, 2, MPI_DOUBLE, MPI_SUM, worker_comm);

             alpha = r_dot_r / p_dot_q_q_dot_q[0];
            beta = alpha * (p_dot_q_q_dot_q[1] / p_dot_q_q_dot_q[0]) - 1;
            r_dot_r = beta * r_dot_r;

            for (int i = 0; i < block_length_separated; ++i)
            {
                r_local[i] -= alpha * q_local[i];
                x_local[i] += alpha * p_local[i];
                p_local[i] = r_local[i] + beta * p_local[i];
            } */
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* printf("worker %d x_local[0]: %f\n", worker_rank, x_local[0]);
        printf("worker %d x_local[1]: %f\n", worker_rank, x_local[1]); */

        // Beware the location of x_local
        MPI_Send(x_local, block_length_separated, MPI_DOUBLE, MASTER, RESULT_TAG, MPI_COMM_WORLD);

        free(A_ij);
        free(x_local);
        free(r_local);
        free(p_local);
        free(q_local);
        free(p_block);
        free(q_block);
        free(q_block_folded);
        /* free(p_dot_q_q_dot_q); */
        free(q_local_transposed);
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();
    return 0;
}
