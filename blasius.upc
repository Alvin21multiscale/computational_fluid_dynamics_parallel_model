#include <string.h>
#include <stdio.h>
#include <math.h>
#include <upc.h>

#define F_Velocity      1
#define Lx              0.3048
#define Ly              0.03
#define nx              1000
#define ny              1000
#define delta_t         0.000001
#define cellsize_x      (Lx / nx)
#define cellsize_y      (Ly / ny)

#define BLOCK_SIZE      (((nx + 2)%THREADS == 0) ? (nx+2)/THREADS*(ny+2) : ((nx+2)/THREADS + 1)*(ny+2))
    
#define sqr(x)          ((x)*(x))

#define DECL(a)                 shared [BLOCK_SIZE] double a[nx+2][ny+2]
#define DECL_STATIC(a)          static shared [BLOCK_SIZE] double a[nx+2][ny+2]
#define FOR(i, n)               for (int i = 0; i < n; ++i)
#define ALL_ROWS(i)             FOR(i, nx+2)
#define ALL_COLS(j)             FOR(j, ny+2)
#define FILL(arr, value)        ALL_ROWS(i) ALL_COLS(j) arr[i][j] = value
#define COPY(a, b)              ALL_ROWS(i) ALL_COLS(j) a[i][j] = b[i][j]
#define FOR_RANGE(i, s, d, e)   for (int i = s; i <= e; i += d)

#define SCALE_MATRIX(s, a, b)       FOR(i, 4) b[i][0] = (s) * a[i][0]
#define SUM_MATRIX(a, b, c, d, e)   FOR(i, 4) e[i][0] = a[i][0] + b[i][0] + c[i][0] + d[i][0]

clock_t start_time;

DECL_STATIC(u_pre);
DECL_STATIC(v_pre);
DECL_STATIC(u_half);
DECL_STATIC(v_half);
DECL_STATIC(u_now);
DECL_STATIC(v_now);

static shared int thread_start[THREADS], thread_end[THREADS];

void print(DECL(a)) {
    ALL_ROWS(i) {
        ALL_COLS(j) {
            if (a[i][j] == 1 || a[i][j] == 0) {
                printf("%11d ", (int)a[i][j]);
            }
            else if (a[i][j] > 1e-4) {
                printf("%.9f ", a[i][j]); // e.g. 0.168054228
            }
            else if (a[i][j] < 0) {
                printf("%.4e ", a[i][j]); // e.g. 2.42994e-16
            }
            else {
                printf("%.5e ", a[i][j]);  // e.g. -1.5092e-19
            }
        }
        printf("\n");
    }
}

void solverB(double u_E_hat, double C_E_hat, double lamdaE[4][1]) {
    lamdaE[0][0] = u_E_hat - C_E_hat;
    lamdaE[1][0] = u_E_hat;
    lamdaE[2][0] = u_E_hat;
    lamdaE[3][0] = u_E_hat + C_E_hat;
}

void solverC(double C_E_hat, double C_N_hat, double u_E_hat, double u_N_hat,
    double v_E_hat, double v_N_hat, DECL(u), DECL(v), int j, int k,
    double deltaE[4][1], double deltaN[4][1]) {
    // compute for F
    deltaE[2][0] = v[j + 1][k] - v[j][k];
    deltaE[1][0] = ((1.33 - 1) / sqr(C_E_hat)) * (
        u_E_hat * (u[j + 1][k] - u[j][k]) -
        (0.5 * (sqr(u[j + 1][k]) + sqr(v[j + 1][k])) - 0.5 * (sqr(u[j][k]) + sqr(v[j][k]))) +
        v_E_hat*(v[j + 1][k] - v[j][k])
        );
    deltaE[0][0] = (1.0 / (2.0 * C_E_hat)) * (u[j][k] - u[j + 1][k] - C_E_hat * deltaE[1][0]);
    deltaE[3][0] = -(deltaE[0][0] + deltaE[1][0]);

    // compute for G
    deltaN[2][0] = v[j][k + 1] - v[j][k];
    deltaN[1][0] = ((1.33 - 1) / sqr(C_N_hat)) * (
        u_N_hat * (u[j][k + 1] - u[j][k]) -
        (0.5 * (sqr(u[j][k + 1]) + sqr(v[j][k + 1])) - 0.5 * (sqr(u[j][k]) + sqr(v[j][k]))) +
        v_N_hat * (v[j][k + 1] - v[j][k])
        );
    deltaN[0][0] = (1.0 / (2.0 * C_N_hat)) * (u[j][k] - u[j][k + 1] - C_N_hat * deltaN[1][0]);
    deltaN[3][0] = -(deltaN[0][0] + deltaN[1][0]);
}

void solverD(double u_E_hat, double v_E_hat, double h_E_hat, double C_E_hat,
    double u_N_hat, double v_N_hat, double h_N_hat, double C_N_hat,
    double T_N_hat[4][4][1], double T_E_hat[4][4][1]) {

#define INIT(P, a, b, c, d) { P[0][0] = a; P[1][0] = b; P[2][0] = c; P[3][0] = d; } 

    // compute for F
    INIT(T_E_hat[0], 1, u_E_hat - C_E_hat, v_E_hat, h_E_hat - (u_E_hat * C_E_hat))
    INIT(T_E_hat[1], 1, u_E_hat, v_E_hat, (sqr(u_E_hat) + sqr(v_E_hat)) / 2);
    INIT(T_E_hat[2], 0, 0, 1, v_E_hat);
    INIT(T_E_hat[3], 1, u_E_hat + C_E_hat, v_E_hat, h_E_hat + u_E_hat * C_E_hat);

    // compute for G
    INIT(T_N_hat[0], 1, u_N_hat - C_N_hat, v_N_hat, h_N_hat - (v_N_hat * C_N_hat));
    INIT(T_N_hat[1], 1, u_N_hat, v_N_hat, (sqr(u_N_hat) + sqr(v_N_hat)) / 2);
    INIT(T_N_hat[2], 0, 0, 1, u_N_hat);
    INIT(T_N_hat[3], 1, u_N_hat + C_N_hat, v_N_hat, h_N_hat + (v_N_hat * C_N_hat));

#undef INIT
}

void main_solver(DECL(u), DECL(v), int j, int k, double *u_E, double *u_square,
    double *u_v, double *v_N, double *v_square, double *v_u) {

    // doing the A term
    double RHO = 1000; // density of water

    // Compute for F
    double velx_E_left = u[j][k] + 0.25*(2.0 / 3 * (u[j][k] - u[j - 1][k]) + 4.0 / 3 * (u[j + 1][k] - u[j][k]));
    double velx_E_right = u[j + 1][k] - 0.25*(2.0 / 3 * (u[j + 2][k] - u[j + 1][k]) + 4.0 / 3 * (u[j + 1][k] - u[j][k]));

    double vely_E_left = v[j][k] + 0.25*(2.0 / 3 * (v[j][k] - v[j - 1][k]) + 4.0 / 3 * (v[j + 1][k] - v[j][k]));
    double vely_E_right = v[j + 1][k] - 0.25*(2.0 / 3 * (v[j + 2][k] - v[j + 1][k]) + 4.0 / 3 * (v[j + 1][k] - v[j][k]));

    double u_E_hat = (sqrt(RHO) * velx_E_left + sqrt(RHO) * velx_E_right) / (2 * sqrt(RHO));
    double v_E_hat = (sqrt(RHO) * vely_E_left + sqrt(RHO) * vely_E_right) / (2 * sqrt(RHO));

    // doing the energy things
    double V_square_E_left = velx_E_left * velx_E_left + vely_E_left * vely_E_left;
    double V_square_E_right = velx_E_right * velx_E_right + vely_E_right * vely_E_right;

    double h_E_left = (0.5 * V_square_E_left) + 403.03;
    double h_E_right = (0.5 * V_square_E_right) + 403.03;

    double h_E_hat = (sqrt(RHO) * h_E_left + sqrt(RHO) * h_E_right) / (2 * sqrt(RHO));

    // doing some dump things of ROE
    double C_E_hat = sqrt((1.33 - 1) * (h_E_hat - 0.5*(sqr(u_E_hat) + sqr(v_E_hat))));

    // Compute for G
    double velx_N_left = u[j][k] + 1.0 / 4 * ((2.0 / 3 * (u[j][k] - u[j][k - 1])) + (4.0 / 3 * (u[j][k + 1] - u[j][k])));
    double velx_N_right = u[j][k + 1] - 1.0 / 4 * ((2.0 / 3 * (u[j][k + 2] - u[j][k + 1])) + (4.0 / 3 * (u[j][k + 1] - u[j][k])));

    double vely_N_left = v[j][k] + 1.0 / 4 * ((2.0 / 3 * (v[j][k] - v[j][k - 1])) + (4.0 / 3 * (v[j][k + 1] - v[j][k])));
    double vely_N_right = v[j][k + 1] - 1.0 / 4 * ((2.0 / 3 * (v[j][k + 2] - v[j][k + 1])) + (4.0 / 3 * (v[j][k + 1] - v[j][k])));

    double v_N_hat = (sqrt(RHO) * vely_N_left + sqrt(RHO) * vely_N_right) / (2 * sqrt(RHO));
    double u_N_hat = (sqrt(RHO) * velx_N_left + sqrt(RHO) * velx_N_right) / (2 * sqrt(RHO));

    // doing the energy things
    double V_square_N_left = velx_N_left * velx_N_left + vely_N_left * vely_N_left;
    double V_square_N_right = velx_N_right * velx_N_right + vely_N_right * vely_N_right;

    double h_N_left = 0.5 * V_square_N_left + 403.03;
    double h_N_right = 0.5 * V_square_N_right + 403.03;

    double h_N_hat = (sqrt(RHO) * h_N_left + sqrt(RHO) * h_N_right) / (2 * sqrt(RHO));

    // doing some dump things of ROE
    double C_N_hat = sqrt((1.33 - 1) * (h_N_hat - (0.5 * (sqr(u_N_hat) + sqr(v_N_hat)))));

    // doing the B term
    double lamdaE[4][1];
    double lamdaN[4][1];
    solverB(u_E_hat, C_E_hat, lamdaE);
    solverB(v_N_hat, C_N_hat, lamdaN);

    // doing the D term
    double T_E_hat[4][4][1];
    double T_N_hat[4][4][1];
    solverD(u_E_hat, v_E_hat, h_E_hat, C_E_hat, u_N_hat, v_N_hat, h_N_hat, C_N_hat, T_E_hat, T_N_hat);

    // doing the C term
    double deltaE[4][1];
    double deltaN[4][1];
    solverC(C_E_hat, C_N_hat, u_E_hat, u_N_hat, v_E_hat, v_N_hat, u, v, j, k, deltaE, deltaN);

    // doing the E term
    // Calculate F_IE
    double matrix_E[4][4][1];
    SCALE_MATRIX(fabs(lamdaE[0][0]) * deltaE[0][0], T_E_hat[0], matrix_E[0]);
    SCALE_MATRIX(fabs(lamdaE[1][0]) * deltaE[1][0], T_E_hat[1], matrix_E[1]);
    SCALE_MATRIX(fabs(lamdaE[2][0]) * deltaE[2][0], T_E_hat[2], matrix_E[2]);
    SCALE_MATRIX(fabs(lamdaE[3][0]) * deltaE[3][0], T_E_hat[3], matrix_E[3]);

    double AL_E[4][1];
    SUM_MATRIX(matrix_E[0], matrix_E[1], matrix_E[2], matrix_E[3], AL_E);
    SCALE_MATRIX(0.5, AL_E, AL_E);

    // calculate for u
    *u_E = 0.5 * (velx_E_left + velx_E_right) - AL_E[0][0];
    *u_square = 0.5 * (*u_E) * (velx_E_left + velx_E_right) - AL_E[1][0];
    *u_v = 0.5 * (velx_E_left * vely_E_left + velx_E_right * vely_E_right) - AL_E[2][0];
    //double uH_E = (0.5 * u_E * (h_E_left + h_E_right)) - AL_E[3][0];

    // Calculate G_IE
    double matrix_N[4][4][1];
    SCALE_MATRIX(fabs(lamdaN[0][0]) * deltaN[0][0], T_N_hat[0], matrix_N[0]);
    SCALE_MATRIX(fabs(lamdaN[1][0]) * deltaN[1][0], T_N_hat[1], matrix_N[1]);
    SCALE_MATRIX(fabs(lamdaN[2][0]) * deltaN[2][0], T_N_hat[2], matrix_N[2]);
    SCALE_MATRIX(fabs(lamdaN[3][0]) * deltaN[3][0], T_N_hat[3], matrix_N[3]);

    double AL_N[4][1];
    SUM_MATRIX(matrix_N[0], matrix_N[1], matrix_N[2], matrix_N[3], AL_N);
    SCALE_MATRIX(0.5, AL_N, AL_N);

    // calculate for v
    *v_N = 0.5 * (vely_N_left + vely_N_right) - AL_N[0][0];
    *v_square = 0.5 * (*v_N) * (vely_N_left + vely_N_right) - AL_N[2][0];
    *v_u = 0.5 * (velx_N_left * vely_N_left + velx_N_right * vely_N_right) - AL_N[1][0];
    //double uH_N = (0.5 * v_N * (h_N_left + h_N_right)) - AL_N[3][0];
}

void invisicid_calculation(DECL(u), DECL(v), DECL(u_pre), DECL(v_pre),
    int half, DECL(u_updated), DECL(v_updated)) {

    DECL_STATIC(uu);      DECL_STATIC(uv);
    DECL_STATIC(vv);      DECL_STATIC(vu);

    upc_forall(int t = 0; t < THREADS; ++t; t) {
        for (int j = thread_start[t]; j < thread_end[t]; ++j) {
            ALL_COLS(k) {
                uu[j][k] = uv[j][k] =
                vv[j][k] = vu[j][k] =
                u_updated[j][k] = v_updated[j][k] = 0;
            }
        }
    }

    upc_barrier;

    upc_forall(int t = 0; t < THREADS; ++t; t) {
        for (int j = thread_start[t]; j < thread_end[t]; ++j) {
            if (j == 0 || j > nx - 1) continue;
            
            FOR_RANGE(k, 2, 1, ny - 1) {
                double a1, a2, a3, a4, a5, a6;
                main_solver(u, v, j, k, &a1, &a2, &a3, &a4, &a5, &a6);
                uu[j][k] = a2;
                uv[j][k] = a3;
                vv[j][k] = a5;
                vu[j][k] = a6;

                /*printf("%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n",
                        ue[j][k], uu[j][k], uv[j][k],
                        vn[j][k], vv[j][k], vu[j][k]);*/
            }
        }
    }
    
    upc_barrier;

    upc_forall(int t = 0; t < THREADS; ++t; t) {
        for (int j = thread_start[t]; j < thread_end[t]; ++j) {
            if (j == 0 || j > nx - 1) continue;
 
            FOR_RANGE(k, 2, 1, ny - 1) {
                double uu_diff = (uu[j][k] - uu[j-1][k]) / cellsize_x ;
                double uv_diff = (uv[j][k] - uv[j-1][k]) / cellsize_x;
                double vv_diff = (vv[j][k] - vv[j][k-1]) / cellsize_y;
                double vu_diff = (vu[j][k] - vu[j][k-1]) / cellsize_y;

                double ux = 1e-6 * ((u[j + 1][k] - 2 * u[j][k] + u[j - 1][k]) / (cellsize_x * cellsize_x));
                double vx = 1e-6 * ((v[j + 1][k] - 2 * v[j][k] + v[j - 1][k]) / (cellsize_x * cellsize_x));
                double uy = 1e-6 * ((u[j][k + 1] - 2 * u[j][k] + u[j][k - 1]) / (cellsize_y * cellsize_y));
                double vy = 1e-6 * ((v[j][k + 1] - 2 * v[j][k] + v[j][k - 1]) / (cellsize_y * cellsize_y));

                if (half == 0) {
                    u_updated[j][k] = (ux + uy - uu_diff - vu_diff) * 0.5 * delta_t + u_pre[j][k];
                    v_updated[j][k] = (vx + vy - uv_diff - vv_diff) * 0.5 * delta_t + v_pre[j][k];
                }
                else {
                    u_updated[j][k] = (ux + uy - uu_diff - vu_diff) * delta_t + u_pre[j][k];
                    v_updated[j][k] = (vx + vy - uv_diff - vv_diff) * delta_t + v_pre[j][k];
                }
            }
        }
    }

    upc_barrier;
    
    upc_forall(int t = 0; t < THREADS; ++t; t) {
        for (int i = thread_start[t]; i < thread_end[t]; ++i) {
            u_updated[i][0] = u_updated[i][1] = 0;
            u_updated[i][ny] = u_updated[i][ny + 1] = F_Velocity;
            v_updated[i][ny] = v_updated[i][ny + 1] = F_Velocity;
        }

        if (MYTHREAD == 0) {
            ALL_COLS(j) u_updated[0][j] = u_updated[1][j] = F_Velocity;
            u_updated[1][0] = 0;
        }

        if (MYTHREAD == THREADS - 1) {
            ALL_COLS(j) u_updated[nx][j] = u_updated[nx + 1][j] = F_Velocity;
            u_updated[nx][0] = 0;
        }
    }

    upc_barrier;
}

void initialize() {
    upc_forall(int t = 0; t < THREADS; ++t; t) {
        if (MYTHREAD == 0) {
            ALL_COLS(j) u_pre[0][j] = u_pre[1][j] = F_Velocity;
            u_pre[1][0] = 0;
        }

        if (MYTHREAD == THREADS - 1) {
            ALL_COLS(j) u_pre[nx][j] = u_pre[nx + 1][j] = F_Velocity;
            u_pre[nx][0] = 0;
        }

        for (int i = thread_start[t]; i < thread_end[t]; ++i) {
            u_pre[i][ny] = u_pre[i][ny + 1] = F_Velocity;
        }
    }

    upc_barrier;
}

int calculate_threads_start_end() {
    int ok = 1;

    if (MYTHREAD == 0) {
        int rows_per_thread = (nx + 2) / THREADS;
        if ((nx + 2) % THREADS > 0) rows_per_thread += 1;
        assert(rows_per_thread * (ny + 2) == BLOCK_SIZE);
        
        FOR(i, THREADS) {
            thread_start[i] = i * rows_per_thread;
            thread_end[i] = (i + 1) * rows_per_thread;
            if (thread_end[i] > nx + 2) {
                thread_end[i] = nx + 2;
            }

            int s = thread_end[i] - thread_start[i];
            if (s <= 1) {
                printf("Thread %d has only %d row. Terminating...\n", i, s);
                ok = 0;
                break;
            }
        }
    }

    upc_barrier;

    return ok;
}

void copy_blasius() {
    upc_forall(int t = 0; t < THREADS; ++t; t) {
        for (int j = thread_start[t]; j < thread_end[t]; ++j) {
            ALL_COLS(k) {
                u_pre[j][k] = u_now[j][k];
                v_pre[j][k] = v_now[j][k];
            }
        }
    }

    upc_barrier;
}

void solve() {    
    FOR_RANGE(t, 0, 1, 10) {
        invisicid_calculation(u_pre, v_pre, u_pre, v_pre, 0, u_half, v_half);

        invisicid_calculation(u_half, v_half, u_pre, v_pre, 1, u_now, v_now);

        copy_blasius();
    }
}

void printToFile(FILE *f, DECL(a)) {
    ALL_ROWS(i) {
        ALL_COLS(j) {
            if (a[i][j] == 1 || a[i][j] == 0) {
                fprintf(f, "%11d ", (int)a[i][j]);
            }
            else if (a[i][j] > 1e-4) {
                fprintf(f, "%.9f ", a[i][j]); // e.g. 0.168054228
            }
            else if (a[i][j] < 0) {
                fprintf(f, "%.4e ", a[i][j]); // e.g. 2.42994e-16
            }
            else {
                fprintf(f, "%.5e ", a[i][j]);  // e.g. -1.5092e-19
            }
        }
        fprintf(f, "\n");
    }
}

void output() {
    if (MYTHREAD == 0) {
        FILE *f = fopen("LastOutput.txt", "w");
        printToFile(f, u_now);
        fprintf(f, "\n");
        printToFile(f, v_now);
        fclose(f);
    }
}

void start_counting_time() {
    if (MYTHREAD == 0) {
        start_time = clock();
    }
}

void print_time() {
    if (MYTHREAD == 0) {
        double time_spent = ((double)clock() - start_time) / CLOCKS_PER_SEC;
        printf("Running time: %G\n", time_spent);
    }
}

int main() {
    start_counting_time();
    if (calculate_threads_start_end()) {
        initialize();
        solve();
        print_time();
        output();
    }
    return 0;
}
