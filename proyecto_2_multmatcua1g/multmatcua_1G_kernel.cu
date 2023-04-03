
/* Para calcular cada fila del tile de C se lee el tile entero de B. Por tanto, para el tile entero de C (el trabajo que hace un bloque de threads) se lee el tile entero de B 
repetidamente (w veces) */

__global__ void sharedABMultiply(float *a, float *b, float *c, int N, const int tile_dim)
{
    extern __shared__ float aTile[];

    float *bTile = aTile + (tile_dim * tile_dim);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    aTile[threadIdx.y * tile_dim + threadIdx.x] = a[row * tile_dim + threadIdx.x];
    bTile[threadIdx.y * tile_dim + threadIdx.x] = b[threadIdx.y * N + col];

    __syncthreads(); // warp usa datos de B leídos por otro warp

    for (int i = 0; i < tile_dim; i++)
    {
        sum += aTile[threadIdx.y * tile_dim + i] * bTile[i * tile_dim + threadIdx.x];
    }
    c[row * N + col] = sum;
}