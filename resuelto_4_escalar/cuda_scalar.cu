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
#include "cuda_scalar_kernel.cu"

void test(float *v, float *w, float *computed, int n)
{
    float *s = (float *)malloc(n * sizeof(float));
    // Multiply and check vectors
    for (int i = 0; i < n; i++)
    {
        s[i] = v[i] * w[i];
        assert(s[i] == computed[i]);
    }

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float *vector_h, *wector_h, *scalar_h, *res; // host data
    float *vector_d, *wector_d, *scalar_d; // device data

    size_t nBytes;

    // default values
    int n = 1;
    int b = 1;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // events
    float processing_time;
    cudaEvent_t start_event, stop_event;

    // process command line arguments
    n = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"n") ? : n;
    b = getCmdLineArgumentInt(argc, (const char **)argv, (const char *)"b") ? : b;

    nBytes = n * sizeof(float);

    // setup execution parameters
    // dim3 grid((n % b) ? (n / b) + 1 : (n / b));
    dim3 grid((n + b - 1) / b);
    dim3 block(b);


    // allocate host memory
    vector_h = (float *)malloc(nBytes);
    wector_h = (float *)malloc(nBytes);
    scalar_h = (float *)malloc(nBytes);
    res = (float *)malloc(sizeof(float));

    for (int i = 0; i < n; i++)
    {
        vector_h[i] = (float)1.0;
        wector_h[i] = (float)2.0;
    }

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **)&vector_d, nBytes));
    checkCudaErrors(cudaMalloc((void **)&wector_d, nBytes));
    checkCudaErrors(cudaMalloc((void **)&scalar_d, nBytes));

    // copy data from host memory to device memory
    checkCudaErrors(cudaMemcpy(vector_d, vector_h, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(wector_d, wector_h, nBytes, cudaMemcpyHostToDevice));

    // execute the kernel
    printf("Running configuration: grid of %d blocks of %d threads (%d threads)\n",
           grid.x, block.x, grid.x * block.x);

    // create events
    checkCudaErrors(cudaEventCreate(&start_event, 0));
    checkCudaErrors(cudaEventCreate(&stop_event, 0));

    // using events
    checkCudaErrors(cudaEventRecord(start_event, 0));

    
    vectorScalarProduct<<<grid, block>>>(vector_d, wector_d, scalar_d, res, n);

    // wait for thread completion
    cudaThreadSynchronize();

    // ///*using event*/
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    cudaEventSynchronize(stop_event); // block until the event is actually recorded
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_event, stop_event));
    printf("Processing time: %f (ms)", processing_time);

    checkCudaErrors(cudaMemcpy(scalar_h, scalar_d, nBytes, cudaMemcpyDeviceToHost));

    // check result
    // test(vector_h, wector_h, scalar_h, n);
    assert(res == (float) 2 * n);


    // free memory
    free(vector_h);
    free(wector_h);
    free(scalar_h);
    checkCudaErrors(cudaFree((void *)vector_d));
    checkCudaErrors(cudaFree((void *)wector_d));
    checkCudaErrors(cudaFree((void *)scalar_d));

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
