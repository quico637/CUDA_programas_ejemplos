////////////////////////////////////////////////////////////////////////////////
// vectorScalar kernel
////////////////////////////////////////////////////////////////////////////////

__device__ void vectorReduce(float *vector_d, float *reduce_d, int n)
{
    extern __shared__ int sdata[];

    // global thread ID in thread block
    unsigned int tidb = threadIdx.x;

    // global thread ID in grid
    unsigned int tidg = blockIdx.x * blockDim.x + threadIdx.x;

    // load shared memory
    sdata[tidb] = (tidg < n) ? vector_d[tidg] : 0;

    __syncthreads();

    if (blockDim.x % 2 != 0 && blockDim.x > 1 && tidb == 0)
    {
        atomicAdd(&sdata[0], sdata[blockDim.x - 1]);
    }

    // perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {

        if (tidb < s)
        {
            sdata[tidb] += sdata[tidb + s];
        }

        // hilo 0 varios bloques
        if (s % 2 != 0 && s > 1 && tidb == 0)
        {
            atomicAdd(&sdata[0], sdata[s - 1]);
        }

        __syncthreads();
    }

    // write result for this block to global memory
    if (tidb == 0)
    {
        // reduce_d[blockIdx.x] = sdata[0];
        atomicAdd(reduce_d, sdata[0]);
    }

}

__global__ void vectorScalarProduct(const float *vector_d, const float *wector_d, float *scalar_d, float* res, int n)
{
    
    // global thread ID in grid
    int tidg = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(tidg < n) {
        scalar_d[tidg] = vector_d[tidg] * wector_d[tidg];   
        // scalar_d[tidg] = shared[tidg];
    }

    if(tidg == 0)
        vectorReduce(scalar_d, res, n);

}
