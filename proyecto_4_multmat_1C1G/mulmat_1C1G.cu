/* -------------------------------------------------------------------------- */
/* Project: I Curso de Computación Científica en Clusters                     */
/* Author:  Juan Fernández Peinador                                           */
/* Date:    Marzo de 2010                                                     */
/* Actualizado en Febrero 2021 para cuda 8.0: cudaDeviceReset()		      */
/* -------------------------------------------------------------------------- */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <omp.h>

// includes, project
#include <cuda.h>
#include <cuda_runtime.h>

// ayuda con los ejemplos
// These are CUDA Helper functions for initialization and error checking
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>

////////////////////////////////////////////////////////////////////////////////

// includes, kernels
#include "mulmat_1C1G_kernel.cu"

#define TEST
// #define DEBUG
#define DEBUG_CUDA

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

float * multiply_row(float *A, float *B, float *C, int m, int n, int w, int row)
{

    for (int i = row; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = 0.0f;
            for (int k = 0; k < w; k++)
            {
                C[i * n + j] += A[i * w + k] * B[k * n + j];
            }
            
        }
    }
    return C;
}

void print_matrix(float *m, int t1, int t2)
{
    for(int i = 0; i < t1; i++)
    {
        for(int j = 0; j < t2; j++)
            printf("%f ", m[i * t2 + j]);
        printf("\n");
    }
}


float * multiply(float *A, float *B,  float *res, int m, int n, int w)
{
    float *C =(float*) malloc(m * n * sizeof(float));

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = 0.0f;
            for (int k = 0; k < w; k++)
            {
                C[i * n + j] += A[i * w + k] * B[k * n + j];
            }

            assert(C[i * n + j] - res[i * n + j] <= 1e-3);
            
        }
    }
    return C;
}

void test(float *A, float *B,  float *res, int m, int n, int w)
{

    

    float *host = multiply(A, B, res, m, n, w);


#ifdef DEBUG
    printf("A: \n");
    print_matrix(A, m, w);

    printf("B: \n");
    print_matrix(B, w, n);

    printf("CUDA: \n");
    print_matrix(res, m, n);

    printf("HOST SECUENTIAL\n");
    print_matrix(host, m, n);
#endif
}

int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C; // host data
    float *d_A, *d_B, *d_C; // device data
    size_t size_A, size_B, size_C;
    size_t nBytes_A, nBytes_B, nBytes_C;

    // default values

    int m = 1;
    int n = 1;   // n
    int k = 1;
    int w = 1;
    int f = 1;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // events
    float processing_time;
    cudaEvent_t start_event, stop_event;

    // process command line arguments
    m = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"M") ?: m;
    n = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"N") ?: n;
    k = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"K") ?: k;
    w = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"W") ?: w;
    f = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"F") ?: f;

    assert(m % w == 0);
    assert(n % w == 0);
    assert(k % w == 0);

    size_A = m * k;
    size_B = k * n;
    size_C = m * n;

    nBytes_A = size_A * sizeof(float);
    nBytes_B = size_B * sizeof(float);
    nBytes_C = size_C * sizeof(float);


    int s = m / w;
    int t = n / w;

    // setup execution parameters
    dim3 grid(t, s);
    dim3 block(w, w);

    // allocate host memory
    h_A = (float *)malloc(nBytes_A);
    h_B = (float *)malloc(nBytes_B);
    h_C = (float *)calloc(size_C, sizeof(float));

    for (int i = 0; i < size_A; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < size_B; i++)
    {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **)&d_A, nBytes_A));
    checkCudaErrors(cudaMalloc((void **)&d_B, nBytes_B));
    checkCudaErrors(cudaMalloc((void **)&d_C, nBytes_C));

    // copy data from host memory to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_C, 0, nBytes_C));

    // execute the kernel
    printf("Running configuration: grid of %dx%d blocks of %dx%d threads (%d threads) - M: %d, N: %d, K: %d, W: %d\n",
           grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y, m, n, k, w);

    // create events
    checkCudaErrors(cudaEventCreate(&start_event, 0));
    checkCudaErrors(cudaEventCreate(&stop_event, 0));

    // using events
    checkCudaErrors(cudaEventRecord(start_event, 0));


    sharedABMultiply<<<grid, block, 2 * w * w * sizeof(float)>>>(d_A, d_B, d_C, m, n, k, w, f);

    // wait for thread completion
    cudaThreadSynchronize();

    // ///*using event*/
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    cudaEventSynchronize(stop_event); // block until the event is actually recorded
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_event, stop_event));
    printf("Processing time: %f (ms)\n", processing_time);

    checkCudaErrors(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));

    multiply_row(h_A, h_B, h_C, m, n, k, f);

// #pragma omp parallel 
// {
//     printf("Hello World... from thread = %d\n", 
//            omp_get_thread_num());
// }  


#ifdef DEBUG_CUDA
    printf("DEBUG CUDA!!! ----------- \n\n");

    printf("A: \n");
    print_matrix(A, m, w);

    printf("B: \n");
    print_matrix(B, w, n);

    printf("CUDA: \n");
    print_matrix(res, m, n);

    printf("HOST SECUENTIAL\n");
    print_matrix(host, m, n);
#endif

#ifdef TEST
    // check result
    test(h_A, h_B, h_C, m, n, k);
#endif
    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree((void *)d_A));
    checkCudaErrors(cudaFree((void *)d_B));
    checkCudaErrors(cudaFree((void *)d_C));

    printf("\nTest PASSED\n");

    //    cudaThreadExit();

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
