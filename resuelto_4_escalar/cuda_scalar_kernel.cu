////////////////////////////////////////////////////////////////////////////////
// vectorScalar kernel
////////////////////////////////////////////////////////////////////////////////

__global__ void vectorScalarProduct(float *vector_d, float *wector_d, float *scalar_d, int n)
{
    extern __shared__ int sdata[];

    printf("legué");

    // global thread ID in thread block
    unsigned int tidb = threadIdx.x;
    
    // global thread ID in grid
    unsigned int tidg = blockIdx.x * blockDim.x + threadIdx.x;


	//printf("blockIdx.x=%d threadIdx.x=%d \n",blockIdx.x,threadIdx.x);

    // load shared memory
    sdata[tidb] = (tidg < n) ? vector_d[tidg]: 1.0;
     
    sdata[tidg] = vector_d[tidg] * wector_d[tidg];

    scalar_d[tidg] = sdata[tidb];    

}
