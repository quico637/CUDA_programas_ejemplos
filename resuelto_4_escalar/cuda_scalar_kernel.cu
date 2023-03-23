////////////////////////////////////////////////////////////////////////////////
// vectorScalar kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void vectorScalarProduct(float *vector_d, float *wector_d, float *scalar_d, int n)
{
    extern __shared__ float shared[];
        
    // global thread ID in grid
    int tidg = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(tidg < n) {
        shared[tidg] = v_shared[tidg] * w_shared[tidg];   
        scalar_d[tidg] = shared[tidg];
    }


}
