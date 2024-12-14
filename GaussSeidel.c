#include <stdio.h>
#include <string.h>
#include <math.h>

void *GaussSeidel(int x, int y, double *A_N, double *A_W, double *A_C, double *A_E, double *A_S, double *phi, double *phi_old, double *p_x, double lambda, int max_iters, double tol){

    for(int iters = 0; iters < max_iters; iters++){
        double max_residual = 0.0;  // To track the max residual

        // Gauss-Seidel iteration
        for(int i = 1; i < (y - 1); i++){
            for(int j = 1; j < (x - 1); j++){
                // Update phi using Gauss-Seidel formula
                phi[i * y + j] = ((lambda / A_C[i * y + j]) * (p_x[i * y + j] 
                    - (A_N[i * y + j] * phi[(i - 1) * y + j] 
                    + A_W[i * y + j] * phi[(i * y) + j - 1] 
                    + A_E[i * y + j] * phi[(i * y) + j + 1] 
                    + A_S[i * y + j] * phi[(i + 1) * y + j]))) 
                    + (1 - lambda) * phi_old[i * y + j]; 
            }
        }
    }

}
