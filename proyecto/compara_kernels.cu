/* -------------------------------------------------------------------------- */
/* Project: I Curso de Computación Científica en Clusters                     */
/* Author:  Juan Fernández Peinador                                           */
/* Date:    Marzo de 2010                                                     */
/* Actualizado en Febrero 2021 para cuda 8.0: cudaDeviceReset()		          */
/* -------------------------------------------------------------------------- */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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
#include "compara_kernels_kernel.cu"

#define TEST
// #define DEBUG

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

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

            // assert(C[i * n + j] == res[i * n + j]);
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
    size_t size_AB, size_C;
    size_t nBytes_AB, nBytes_C;

    // default values
    int dim_mat = 1;   // n
    int dim_block = 1; // w
    int kernel = 1;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // events
    float processing_time;
    cudaEvent_t start_event, stop_event;

    // process command line arguments
    dim_mat = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"N") ?: dim_mat;
    dim_block = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"W") ?: dim_block;
    kernel = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"K") ?: kernel;

    assert(dim_mat % dim_block == 0);

    size_AB = dim_mat * dim_block;
    size_C = dim_mat * dim_mat;

    nBytes_AB = size_AB * sizeof(float);
    nBytes_C = size_C * sizeof(float);

    // setup execution parameters
    dim3 grid(dim_mat / dim_block, dim_mat / dim_block);
    dim3 block(dim_block, dim_block);

    // allocate host memory
    h_A = (float *)malloc(nBytes_AB);
    h_B = (float *)malloc(nBytes_AB);
    h_C = (float *)calloc(size_C, sizeof(float));

    for (int i = 0; i < size_AB; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **)&d_A, nBytes_AB));
    checkCudaErrors(cudaMalloc((void **)&d_B, nBytes_AB));
    checkCudaErrors(cudaMalloc((void **)&d_C, nBytes_C));

    // copy data from host memory to device memory
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes_AB, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, nBytes_AB, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_C, 0, nBytes_C));

    // execute the kernel
    // printf("Running configuration: grid of %dx%d blocks of %dx%d threads (%d threads) - KERNEL: %d\n",
    //        grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y, kernel);

    // create events
    checkCudaErrors(cudaEventCreate(&start_event, 0));
    checkCudaErrors(cudaEventCreate(&stop_event, 0));

    // using events
    checkCudaErrors(cudaEventRecord(start_event, 0));

    switch (kernel)
    {
    case 1:
        // printf("dim_block: %d", dim_block);
        simpleMultiply<<<grid, block>>>(d_A, d_B, d_C, dim_mat, dim_block);
        break;

    case 2:
        coalescedMultiply<<<grid, block, dim_block * dim_block * sizeof(float)>>>(d_A, d_B, d_C, dim_mat, dim_block);
        break;

    case 3:
        sharedABMultiply<<<grid, block, 2 * dim_block * dim_block * sizeof(float)>>>(d_A, d_B, d_C, dim_mat, dim_block);
        break;

    default:
        printf("No kernel found for that index, please try with a number between [1,3]");
        exit(1);
        break;
    }

    // wait for thread completion
    cudaThreadSynchronize();

    // ///*using event*/
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    cudaEventSynchronize(stop_event); // block until the event is actually recorded
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_event, stop_event));
    // printf("Processing time: %f (ms)\n", processing_time);

    checkCudaErrors(cudaMemcpy(h_C, d_C, nBytes_C, cudaMemcpyDeviceToHost));

    // omen

#ifdef TEST
    // check result
    test(h_A, h_B, h_C, dim_mat, dim_mat, dim_block);
#endif
    // free memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree((void *)d_A));
    checkCudaErrors(cudaFree((void *)d_B));
    checkCudaErrors(cudaFree((void *)d_C));

    // printf("\nTest PASSED\n");

    printf("%d;%d;%d;%.2lf\n", kernel, dim_mat, dim_block, processing_time);

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
