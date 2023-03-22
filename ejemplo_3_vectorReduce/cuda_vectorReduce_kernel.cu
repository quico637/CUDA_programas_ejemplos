////////////////////////////////////////////////////////////////////////////////
// vectorReduce kernel
////////////////////////////////////////////////////////////////////////////////

__device__ int find_geq_two_power(unsigned int n)
{
    unsigned int i = 0;
    while(i < n)
    {
        i *= 2
    }
    return i;
}

__global__ void vectorReduce(float *vector_d, float *reduce_d, int n)
{
    extern __shared__ int sdata[];

    // global thread ID in thread block
    unsigned int tidb = threadIdx.x;
    
    // global thread ID in grid
    unsigned int tidg = blockIdx.x * blockDim.x + threadIdx.x;


	//printf("blockIdx.x=%d threadIdx.x=%d \n",blockIdx.x,threadIdx.x);

    // load shared memory
    sdata[tidb] = (tidg < n) ? vector_d[tidg]: 0;

    __syncthreads();
     
    // perform reduction in shared memory

    unsigned int next_two_power = find_geq_two_power(blockIdx.x);

    for(unsigned int s = next_two_power/2; s > 0; s = next_two_power(s << 1)) {
        if (tidb < s) {
            sdata[tidb] += sdata[tidb + s];
        }
        __syncthreads();
    }

                        
    // write result for this block to global memory
    if (tidb == 0) {
        reduce_d[blockIdx.x] = sdata[0];
    }
}
