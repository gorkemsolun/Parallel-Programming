// Görkem Kadir Solun 22003214
// MPI‐parallel CG for dense SPD; master (rank 0) only does I/O & final gather.
// P2P for p‐exchange before each SpMV, collective only for inner products.
// Follows Algorithm 3 of Selvitopi et al.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MASTER 0
#define TAG_DIST  100
#define TAG_PSEND 200
#define TAG_XSEND 300

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == MASTER) {
            fprintf(stderr, "Need ≥2 MPI ranks\n");
        }
        MPI_Finalize(); return 1;
    }

    // Split out a communicator for all the workers (ranks 1..size-1)
    MPI_Comm worker_comm;
    MPI_Comm_split(MPI_COMM_WORLD,
        world_rank == MASTER ? MPI_UNDEFINED : 1,
        world_rank,
        &worker_comm);

    int n, rows_per;
    double* A = NULL, * b = NULL;

    if (world_rank == MASTER) {
        //─── MASTER: read input and distribute ────────────────────────────────
        FILE* fin = fopen("input.txt", "r");
        if (!fin) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fscanf(fin, "%d", &n);
        rows_per = n / (world_size - 1);
        A = malloc(n * n * sizeof(double));
        b = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0;j < n;j++) {
                fscanf(fin, "%lf", &A[i * n + j]);
            }
        }
        for (int i = 0;i < n;i++) fscanf(fin, "%lf", &b[i]);
        fclose(fin);

        // tell each worker n & rows_per, then its slice of A & b
        for (int r = 1;r < world_size;r++) {
            MPI_Send(&n, 1, MPI_INT, r, TAG_DIST, MPI_COMM_WORLD);
            MPI_Send(&rows_per, 1, MPI_INT, r, TAG_DIST, MPI_COMM_WORLD);
            int offset = (r - 1) * rows_per;
            MPI_Send(A + offset * n, rows_per * n, MPI_DOUBLE, r, TAG_DIST, MPI_COMM_WORLD);
            MPI_Send(b + offset, rows_per, MPI_DOUBLE, r, TAG_DIST, MPI_COMM_WORLD);
            fprintf(stderr,
                "[Master] Sent rows %d..%d to rank %d\n",
                offset, offset + rows_per - 1, r);
        }
        double t_start = MPI_Wtime();

        //─── MASTER: collect final x ─────────────────────────────────────────
        double* x = malloc(n * sizeof(double));
        for (int r = 1;r < world_size;r++) {
            int offset = (r - 1) * rows_per;
            MPI_Recv(x + offset, rows_per, MPI_DOUBLE, r, TAG_XSEND,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            fprintf(stderr, "[Master] Received x[%d..%d] from rank %d\n",
                offset, offset + rows_per - 1, r);
        }
        double t_end = MPI_Wtime();
        fprintf(stderr, "[Master] Total CG time: %.6f s\n", t_end - t_start);

        // write output
        FILE* fout = fopen("p_output.txt", "w");
        for (int i = 0; i < n; i++) {
            fprintf(fout, "%.8f\n", x[i]);
        }
        fclose(fout);

        free(A); free(b); free(x);
    } else {
        //─── WORKER: receive input slice ───────────────────────────────────────
        MPI_Recv(&n, 1, MPI_INT, MASTER, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows_per, 1, MPI_INT, MASTER, TAG_DIST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double* A_loc = malloc(rows_per * n * sizeof(double));
        double* b_loc = malloc(rows_per * sizeof(double));
        MPI_Recv(A_loc, rows_per * n, MPI_DOUBLE, MASTER, TAG_DIST,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_loc, rows_per, MPI_DOUBLE, MASTER, TAG_DIST,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "[Rank %d] Got %d×%d block\n", world_rank, rows_per, n);

        // build worker_comm params
        int wrank, wsize;
        MPI_Comm_rank(worker_comm, &wrank);
        MPI_Comm_size(worker_comm, &wsize);

        // allocate CG vectors
        double* x_loc = calloc(rows_per, sizeof(double));
        double* r_loc = malloc(rows_per * sizeof(double));
        double* p_loc = malloc(rows_per * sizeof(double));
        double* q_loc = malloc(rows_per * sizeof(double));
        // initial r=b, p=r
        for (int i = 0;i < rows_per;i++) {
            r_loc[i] = b_loc[i];
            p_loc[i] = r_loc[i];
        }

        // full p and r will live here
        double* p_full = malloc(n * sizeof(double));

        const double tol = 1e-8;
        double rdot, pdotq, rdot_new, alpha, beta;
        int maxIter = n;

        for (int iter = 0;iter < maxIter;iter++) {
            //─── 1) P2P EXCHANGE of p among workers ──────────────────────
            // place my chunk into the right spot
            int my_offset = wrank * rows_per;
            for (int i = 0;i < rows_per;i++) p_full[my_offset + i] = p_loc[i];
            // post receives from all peers, then sends
            MPI_Request* reqs = malloc(2 * (wsize - 1) * sizeof(MPI_Request));
            int reqc = 0;
            for (int peer = 0;peer < wsize;peer++) {
                if (peer == wrank) continue;
                int peer_offset = peer * rows_per;
                MPI_Irecv(p_full + peer_offset, rows_per, MPI_DOUBLE,
                    peer + 1, TAG_PSEND, MPI_COMM_WORLD,
                    &reqs[reqc++]);
            }
            for (int peer = 0;peer < wsize;peer++) {
                if (peer == wrank) continue;
                MPI_Isend(p_loc, rows_per, MPI_DOUBLE,
                    peer + 1, TAG_PSEND, MPI_COMM_WORLD,
                    &reqs[reqc++]);
            }
            MPI_Waitall(reqc, reqs, MPI_STATUSES_IGNORE);
            free(reqs);
            fprintf(stderr, "[Rank %d] iter %d: p exchange done\n", world_rank, iter);

            //─── 2) local SpMV q_loc = A_loc * p_full ──────────────────
            for (int i = 0;i < rows_per;i++) {
                double sum = 0;
                for (int j = 0;j < n;j++) {
                    sum += A_loc[i * n + j] * p_full[j];
                }
                q_loc[i] = sum;
            }
            fprintf(stderr, "[Rank %d] iter %d: SpMV done\n", world_rank, iter);

            //─── 3) local dot‐products ─────────────────────────────────
            pdotq = 0;
            rdot = 0;
            for (int i = 0;i < rows_per;i++) {
                pdotq += p_loc[i] * q_loc[i];
                rdot += r_loc[i] * r_loc[i];
            }

            //─── 4) collective reductions among workers ────────────────
            double sum_pq, sum_rr;
            MPI_Allreduce(&pdotq, &sum_pq, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
            MPI_Allreduce(&rdot, &sum_rr, 1, MPI_DOUBLE, MPI_SUM, worker_comm);
            fprintf(stderr, "[Rank %d] iter %d: reduce pq=%.6e rr=%.6e\n",
                world_rank, iter, sum_pq, sum_rr);

            if (iter == 0) {
                alpha = sum_rr / sum_pq;
            } else {
                beta = sum_rr / rdot;
                alpha = sum_rr / sum_pq;
            }

            //─── 5) update x, r, p ───────────────────────────────────
            for (int i = 0;i < rows_per;i++) {
                x_loc[i] += alpha * p_loc[i];
                r_loc[i] -= alpha * q_loc[i];
            }

            // prepare for next iter
            rdot_new = 0;
            for (int i = 0;i < rows_per;i++) rdot_new += r_loc[i] * r_loc[i];
            MPI_Allreduce(MPI_IN_PLACE, &rdot_new, 1, MPI_DOUBLE, MPI_SUM, worker_comm);

            if (sqrt(rdot_new) < tol) {
                if (wrank == 0) fprintf(stderr, "[Rank %d] Converged at iter %d\n", world_rank, iter);
                break;
            }

            beta = rdot_new / sum_rr;
            for (int i = 0;i < rows_per;i++) {
                p_loc[i] = r_loc[i] + beta * p_loc[i];
            }
            rdot = rdot_new;
        }

        //─── WORKER: send final x_loc back to master ───────────────────
        MPI_Send(x_loc, rows_per, MPI_DOUBLE,
            MASTER, TAG_XSEND, MPI_COMM_WORLD);

        // cleanup
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
