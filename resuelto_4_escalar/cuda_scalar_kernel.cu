////////////////////////////////////////////////////////////////////////////////
// vectorScalar kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void vectorScalarProduct(float *vector_d, float *wector_d, float *scalar_d, int n)
{
    extern __shared__ float v_shared[];
    extern __shared__ float w_shared[];
    extern __shared__ float s_shared[];

    // global thread ID in thread block
    int tidb = threadIdx.x;
    
    // global thread ID in grid
    int tidg = blockIdx.x * blockDim.x + threadIdx.x;


	//printf("blockIdx.x=%d threadIdx.x=%d \n",blockIdx.x,threadIdx.x);

    // load shared memory
    v_shared[tidb] = (tidg < n) ? vector_d[tidg] : 1.0;
    w_shared[tidb] = (tidg < n) ? wector_d[tidg] : 1.0;
     
    s_shared[tidg] = v_shared[tidg] * w_shared[tidg];

    scalar_d[tidg] = s_shared[tidb];

}
