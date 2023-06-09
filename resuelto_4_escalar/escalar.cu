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
#include "escalar_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float *vector_h, *wector_h, *scalar_h, *reduce_h; // host data
    float *vector_d, *wector_d, *scalar_d, *reduce_d; // device data
    size_t nBytes;

    // default values
    int n = 1;
    int bsx = 1;


 // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;


    //events
    float processing_time;
    cudaEvent_t start_event, stop_event;	


    // process command line arguments
    n=getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "n")?:n;
    bsx=getCmdLineArgumentInt(argc, (const char **) argv, (const char *) "b")?:bsx;

    nBytes = n * sizeof(float);

    // setup execution parameters
    dim3 grid( (n%bsx) ? (n/bsx)+1 : (n/bsx) );
    dim3 block(bsx);

    // allocate host memory
    vector_h = (float *) malloc(nBytes);
    wector_h = (float *) malloc(nBytes);
    scalar_h = (float *) malloc(nBytes);

    for(int i = 0; i < n; i++)
    {
        // vector_h[i] = (float) 1.0;
        // wector_h[i] = (float) 2.0;

        vector_h[i] = rand()/(float)RAND_MAX;
        wector_h[i] = rand()/(float)RAND_MAX;
    }
    reduce_h = (float *) malloc(sizeof(float));
    bzero(reduce_h, 1 * sizeof(float));
    
    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &vector_d, nBytes));
    checkCudaErrors(cudaMalloc((void **) &wector_d, nBytes));
    checkCudaErrors(cudaMalloc((void **) &scalar_d, nBytes));
    checkCudaErrors(cudaMalloc((void **) &reduce_d, sizeof(float)));

    // copy data from host memory to device memory
    checkCudaErrors(cudaMemcpy(vector_d, vector_h, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(wector_d, wector_h, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(reduce_d, 0, sizeof(float)));

    // execute the kernel
    printf("Running configuration: grid of %d blocks of %d threads (%d threads)\n", 
           grid.x, block.x, grid.x * block.x );


    //create events
    checkCudaErrors(cudaEventCreate(&start_event,0));
    checkCudaErrors(cudaEventCreate(&stop_event,0));
    
    //using events
    checkCudaErrors(cudaEventRecord(start_event,0));


    vectorReduce<<<grid, block, block.x * sizeof(float)>>>(vector_d, reduce_d, wector_d, scalar_d, n);
    
    // wait for thread completion
    cudaThreadSynchronize();





 // ///*using event*/        
    checkCudaErrors(cudaEventRecord(stop_event, 0));        
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded        
    checkCudaErrors(cudaEventElapsedTime(&processing_time, start_event, stop_event));        
    printf("Processing time: %f (ms)", processing_time);       


    checkCudaErrors(cudaMemcpy(reduce_h, reduce_d, sizeof(float), cudaMemcpyDeviceToHost));

    // check result
    // assert(*reduce_h == (float) 2 * n);

    float sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum += vector_h[i] * wector_h[i];
    }
    printf("gpu: %lf cpu: %lf", *reduce_h, sum);
    assert(*reduce_h - sum <= 1e-3);

    // free memory
    free(vector_h);
    free(wector_h);
    free(scalar_h);
    free(reduce_h);
    checkCudaErrors(cudaFree((void *) vector_d));
    checkCudaErrors(cudaFree((void *) wector_d));
    checkCudaErrors(cudaFree((void *) scalar_d));
    checkCudaErrors(cudaFree((void *) reduce_d));

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
