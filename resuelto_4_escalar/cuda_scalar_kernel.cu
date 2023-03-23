////////////////////////////////////////////////////////////////////////////////
// vectorScalar kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void vectorScalarProduct(const float *vector_d, const float *wector_d, const float *scalar_d, int n)
{
    
    // global thread ID in grid
    int tidg = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(tidg < n) {
        scalar_d[tidg] = vector_d[tidg] * wector_d[tidg];   
        // scalar_d[tidg] = shared[tidg];
    }


}
