// Görkem Kadir Solun 22003214
// MPI‐parallel CG for dense SPD; master (rank 0) only does I/O & final gather.
// P2P for p‐exchange before each SpMV, collective only for inner products.
// Follows Algorithm 3 of Selvitopi et al.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0
#define DISTRIBUTE_TAG  100
#define P_SEND_TAG 200
#define RESULT_SEND_TAG 300

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2) {
        if (world_rank == MASTER) {
            fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);

        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const char* input_file_name = argv[1];

    if (world_size < 2) {
        if (world_rank == MASTER) {
            fprintf(stderr, "Need >=2 MPI ranks\n");
        }
        MPI_Finalize(); return 1;
    }

    // Split out a communicator for all the workers (ranks 1..size-1)
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
        world_rank == MASTER ? MPI_UNDEFINED : 1,
        world_rank,
        &worker_comm);

    int n, rows_per_worker;
    double* A = NULL, * b = NULL;

    if (world_rank == MASTER) {
        FILE* input_file = fopen(input_file_name, "r");
        if (!input_file) {
            perror("fopen");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(input_file, "%d", &n);

        // Assuming n is divisible by world_size - 1
        rows_per_worker = n / (world_size - 1);
        A = malloc(n * n * sizeof(double));
        b = malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fscanf(input_file, "%lf", &A[i * n + j]);
            }
        }
        for (int i = 0; i < n; ++i) {
            fscanf(input_file, "%lf", &b[i]);
        }
        fclose(input_file);

        for (int w = 1; w < world_size; ++w) {
            MPI_Send(&n, 1, MPI_INT, w, DISTRIBUTE_TAG, MPI_COMM_WORLD);
            MPI_Send(&rows_per_worker, 1, MPI_INT, w, DISTRIBUTE_TAG, MPI_COMM_WORLD);

            int offset = (w - 1) * rows_per_worker;
            MPI_Send(A + offset * n, rows_per_worker * n, MPI_DOUBLE, w, DISTRIBUTE_TAG, MPI_COMM_WORLD);
            MPI_Send(b + offset, rows_per_worker, MPI_DOUBLE, w, DISTRIBUTE_TAG, MPI_COMM_WORLD);

            fprintf(stderr,
                "[Master] Sent rows %d..%d to rank %d\n",
                offset, offset + rows_per_worker - 1, w);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double t_end = MPI_Wtime();
        fprintf(stderr, "[Master] Time after the initial distribution phase and before the final reduction of the output vector: %.6f s\n", t_end - t_start);

        double* result = malloc(n * sizeof(double));
        for (int w = 1; w < world_size; ++w) {
            int offset = (w - 1) * rows_per_worker;
            MPI_Recv(result + offset, rows_per_worker, MPI_DOUBLE, w, RESULT_SEND_TAG,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            fprintf(stderr, "[Master] Received x[%d..%d] from rank %d\n",
                offset, offset + rows_per_worker - 1, w);
        }

        FILE* output_file = fopen("p_output.txt", "w");
        for (int i = 0; i < n; ++i) {
            fprintf(output_file, "%.8f\n", result[i]);
        }
        fclose(output_file);

        free(A);
        free(b);
        free(result);
    } else {
        MPI_Recv(&n, 1, MPI_INT, MASTER, DISTRIBUTE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_per_worker, 1, MPI_INT, MASTER, DISTRIBUTE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double* A_loc = malloc(rows_per_worker * n * sizeof(double));
        double* b_loc = malloc(rows_per_worker * sizeof(double));
        MPI_Recv(A_loc, rows_per_worker * n, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_loc, rows_per_worker, MPI_DOUBLE, MASTER, DISTRIBUTE_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "[Rank %d] Got %d×%d block\n", world_rank, rows_per_worker, n);

        MPI_Barrier(MPI_COMM_WORLD);

        int worker_rank, worker_size;
        MPI_Comm_rank(worker_comm, &worker_rank);
        MPI_Comm_size(worker_comm, &worker_size);

        double* x_loc = calloc(rows_per_worker, sizeof(double));
        double* r_loc = malloc(rows_per_worker * sizeof(double));
        double* p_loc = malloc(rows_per_worker * sizeof(double));
        double* q_loc = malloc(rows_per_worker * sizeof(double));

        // initial r=b, p=r
        for (int i = 0; i < rows_per_worker; ++i) {
            r_loc[i] = b_loc[i];
            p_loc[i] = r_loc[i];
        }

        double r_dot_r_rho = 0;
        for (int i = 0; i < rows_per_worker; ++i) {
            r_dot_r_rho += r_loc[i] * r_loc[i];
        }
        // reduce r_dot_r_rho to all workers
        double r_dot_new;
        MPI_Allreduce(&r_dot_r_rho, &r_dot_new, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
        r_dot_r_rho = r_dot_new;

        fprintf(stderr, "[Rank %d] r_dot_r_rho=%.6e\n", world_rank, r_dot_new);

        // full p and r location
        double* p_full = malloc(n * sizeof(double));

        double q_dot_q_kappa, p_dot_q_pi, alpha, beta;
        for (int iteration = 0; iteration < n; ++iteration) {
            int my_offset = worker_rank * rows_per_worker;
            for (int i = 0; i < rows_per_worker; ++i) {
                p_full[my_offset + i] = p_loc[i];
            }

            // post receives from all peers, then sends
            MPI_Request* requests = malloc(2 * (worker_size - 1) * sizeof(MPI_Request));
            int request_counter = 0;
            for (int peer = 0;peer < worker_size; ++peer) {
                if (peer == worker_rank) {
                    continue;
                }

                int peer_offset = peer * rows_per_worker;
                MPI_Irecv(p_full + peer_offset, rows_per_worker, MPI_DOUBLE,
                    peer + 1, P_SEND_TAG, MPI_COMM_WORLD,
                    &requests[request_counter++]);
            }

            for (int peer = 0; peer < worker_size; ++peer) {
                if (peer == worker_rank) {
                    continue;
                }

                MPI_Isend(p_loc, rows_per_worker, MPI_DOUBLE,
                    peer + 1, P_SEND_TAG, MPI_COMM_WORLD,
                    &requests[request_counter++]);
            }

            MPI_Waitall(request_counter, requests, MPI_STATUSES_IGNORE);
            free(requests);

            fprintf(stderr, "[Rank %d] iteration %d: p exchange done\n", world_rank, iteration);

            // local SpMV q_loc = A_loc * p_full
            for (int i = 0; i < rows_per_worker; ++i) {
                double sum = 0;
                for (int j = 0; j < n; ++j) {
                    sum += A_loc[i * n + j] * p_full[j];
                }
                q_loc[i] = sum;
            }
            fprintf(stderr, "[Rank %d] iteration %d: SpMV done\n", world_rank, iteration);

            // local dot products
            p_dot_q_pi = 0;
            q_dot_q_kappa = 0;
            for (int i = 0; i < rows_per_worker; ++i) {
                p_dot_q_pi += p_loc[i] * q_loc[i];
                q_dot_q_kappa += q_loc[i] * q_loc[i];
            }

            double sum_p_dot_q_pi, sum_q_dot_q_kappa;
            MPI_Allreduce(&p_dot_q_pi, &sum_p_dot_q_pi, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
            MPI_Allreduce(&q_dot_q_kappa, &sum_q_dot_q_kappa, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            fprintf(stderr, "[Rank %d] iteration %d: reduce pq=%.6e rr=%.6e\n",
                world_rank, iteration, sum_p_dot_q_pi, sum_q_dot_q_kappa);


            alpha = r_dot_r_rho / sum_p_dot_q_pi;
            beta = alpha * sum_q_dot_q_kappa / sum_p_dot_q_pi - 1;

            // update x, r, p, r_dot_r_rho
            r_dot_r_rho = beta * r_dot_r_rho;
            for (int i = 0; i < rows_per_worker; ++i) {
                x_loc[i] += alpha * p_loc[i];
                r_loc[i] -= alpha * q_loc[i];
                p_loc[i] = r_loc[i] + beta * p_loc[i];
            }

            if (sqrt(r_dot_new) < 1e-8) {
                if (worker_rank == 0) fprintf(stderr, "[Rank %d] Converged at iteration %d\n", world_rank, iteration);
                break;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Send(x_loc, rows_per_worker, MPI_DOUBLE,
            MASTER, RESULT_SEND_TAG, MPI_COMM_WORLD);

        free(A_loc);
        free(b_loc);
        free(x_loc);
        free(r_loc);
        free(p_loc);
        free(q_loc);
        free(p_full);
        MPI_Comm_free(&worker_comm);
    }

    MPI_Finalize();
    return 0;
}
