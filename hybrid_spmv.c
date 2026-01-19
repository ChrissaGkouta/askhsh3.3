#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h> // Προσθήκη βιβλιοθήκης OpenMP

#define EPSILON 1e-6

// --- Βοηθητικές Συναρτήσεις ---

void convert_to_csr(int *dense, int n, int nnz, int **values, int **col_inds, int **row_ptr) {
    *values = (int *)malloc(nnz * sizeof(int));
    *col_inds = (int *)malloc(nnz * sizeof(int));
    *row_ptr = (int *)malloc((n + 1) * sizeof(int));

    if (!*values || !*col_inds || !*row_ptr) exit(1);

    int count = 0;
    (*row_ptr)[0] = 0;
    
    // Σειριακή κατασκευή (συνήθως γίνεται στο rank 0 μια φορά)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dense[i * n + j] != 0) {
                (*values)[count] = dense[i * n + j];
                (*col_inds)[count] = j;
                count++;
            }
        }
        (*row_ptr)[i + 1] = count;
    }
}

// Υβριδικός SpMV Kernel (OpenMP parallelization inside MPI rank)
void spmv_csr_kernel(int rows, int *val, int *col, int *rpt, double *x, double *y) {
    // Οδηγία OpenMP για παραλληλισμό του βρόχου
    // schedule(dynamic): Καλύτερο για CSR γιατί οι γραμμές μπορεί να έχουν διαφορετικό αριθμό στοιχείων
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        int row_start = rpt[i];
        int row_end = rpt[i+1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += val[j] * x[col[j]];
        }
        y[i] = sum;
    }
}

// Υβριδικός Dense Kernel
void dense_mv_kernel(int rows, int n, int *matrix, double *x, double *y) {
    // schedule(static): Εδώ ο φόρτος είναι σταθερός (n στοιχεία ανά γραμμή)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    int rank, size, provided;
    int n, iterations;
    float sparsity;

    // Ζητάμε υποστήριξη νημάτων, αν και εδώ χρησιμοποιούμε MPI έξω από τα parallel regions
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s <N> <Sparsity> <Iterations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    sparsity = atof(argv[2]);
    iterations = atoi(argv[3]);

    // Timers
    double t_start_total, t_end_total;
    double t_calc_start, t_calc_end;

    // Data Structures
    int *dense_matrix = NULL;
    double *vector_x = (double *)malloc(n * sizeof(double)); 
    double *vector_y_local = NULL; 

    // CSR structures (Global - only at root)
    int *csr_val = NULL, *csr_col = NULL, *csr_row_ptr = NULL;
    int nnz_total = 0;

    // CSR structures (Local)
    int *loc_val = NULL, *loc_col = NULL, *loc_row_ptr = NULL;

    // --- Setup (Rank 0) ---
    int remainder = n % size;
    int local_rows = n / size + (rank < remainder ? 1 : 0);
    
    int *scounts_rows = malloc(size * sizeof(int));
    int *displs_rows = malloc(size * sizeof(int));
    int *scounts_nnz = NULL;
    int *displs_nnz = NULL;

    if (rank == 0) {
        // Εκτύπωση πληροφοριών Hybrid περιβάλλοντος
        #pragma omp parallel 
        {
            #pragma omp master
            printf(">> Hybrid Run: MPI Ranks=%d, OpenMP Threads per Rank=%d\n", size, omp_get_num_threads());
        }

        scounts_nnz = malloc(size * sizeof(int));
        displs_nnz = malloc(size * sizeof(int));
        
        srand(time(NULL));
        dense_matrix = (int *)malloc(n * n * sizeof(int));
        
        int r_sum = 0;
        for(int i=0; i<size; i++) {
            scounts_rows[i] = n / size + (i < remainder ? 1 : 0);
            displs_rows[i] = r_sum;
            r_sum += scounts_rows[i];
        }

        for (int i = 0; i < n; i++) vector_x[i] = 1.0;
        
        for (int i = 0; i < n * n; i++) {
            if ((float)rand() / RAND_MAX > sparsity) {
                dense_matrix[i] = (rand() % 10) + 1;
                nnz_total++;
            } else dense_matrix[i] = 0;
        }
        convert_to_csr(dense_matrix, n, nnz_total, &csr_val, &csr_col, &csr_row_ptr);

        // Υπολογισμός NNZ counts για scatter
        int current_row = 0;
        int nnz_sum = 0;
        for (int i = 0; i < size; i++) {
            int rows_proc = scounts_rows[i];
            int start = csr_row_ptr[current_row];
            int end = csr_row_ptr[current_row + rows_proc];
            scounts_nnz[i] = end - start;
            displs_nnz[i] = nnz_sum;
            nnz_sum += scounts_nnz[i];
            current_row += rows_proc;
        }
    } else {
        // Sync counts
        int r_sum = 0;
        for(int i=0; i<size; i++) {
            scounts_rows[i] = n / size + (i < remainder ? 1 : 0);
            displs_rows[i] = r_sum;
            r_sum += scounts_rows[i];
        }
    }

    vector_y_local = (double *)malloc(local_rows * sizeof(double));

    // --- Διανομή Δεδομένων ---
    MPI_Bcast(vector_x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int my_nnz;
    MPI_Scatter(scounts_nnz, 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_nnz > 0) {
        loc_val = (int *)malloc(my_nnz * sizeof(int));
        loc_col = (int *)malloc(my_nnz * sizeof(int));
    }
    loc_row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));

    MPI_Scatterv(csr_val, scounts_nnz, displs_nnz, MPI_INT, loc_val, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(csr_col, scounts_nnz, displs_nnz, MPI_INT, loc_col, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        int current_r = 0;
        for(int i=0; i<=local_rows; i++) loc_row_ptr[i] = csr_row_ptr[i];
        for(int p=1; p<size; p++) {
            int r_count = scounts_rows[p];
            current_r += scounts_rows[p-1];
            MPI_Send(&csr_row_ptr[current_r], r_count + 1, MPI_INT, p, 99, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(loc_row_ptr, local_rows + 1, MPI_INT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int offset = loc_row_ptr[0];
        for(int i=0; i<=local_rows; i++) loc_row_ptr[i] -= offset;
    }

    // --- Hybrid Calculation Loop ---
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) t_start_total = MPI_Wtime();
    
    // Μετράμε μόνο τον χρόνο υπολογισμού (χωρίς τα setup)
    if(rank == 0) t_calc_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        // Παράλληλη εκτέλεση με OpenMP (εσωτερικά)
        spmv_csr_kernel(local_rows, loc_val, loc_col, loc_row_ptr, vector_x, vector_y_local);
        
        // Επικοινωνία με MPI (Master thread only)
        MPI_Allgatherv(vector_y_local, local_rows, MPI_DOUBLE, 
                       vector_x, scounts_rows, displs_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) {
        t_calc_end = MPI_Wtime();
        t_end_total = MPI_Wtime();
        
        printf("RESULTS: N=%d, Sp=%.2f, Iter=%d, Ranks=%d\n", n, sparsity, iterations, size);
        printf("Time_Hybrid_Calc: %f\n", t_calc_end - t_calc_start);
        printf("Time_Hybrid_Total: %f\n", t_end_total - t_start_total);
    }

    // Cleanup
    free(vector_x); free(vector_y_local);
    if(loc_val) free(loc_val);
    if(loc_col) free(loc_col);
    free(loc_row_ptr); free(scounts_rows); free(displs_rows);
    if(rank==0) {
        free(dense_matrix); free(csr_val); free(csr_col); free(csr_row_ptr);
        free(scounts_nnz); free(displs_nnz);
    }

    MPI_Finalize();
    return 0;
}