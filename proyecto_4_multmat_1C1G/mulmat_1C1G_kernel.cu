/* Para calcular cada fila del tile de C se lee el tile entero de B. Por tanto, para el tile entero de C (el trabajo que hace un bloque de threads) se lee el tile entero de B
repetidamente (w veces) */

__global__ void sharedABMultiply(float *a, float *b, float *c, const int M, const int N, const int K, const int tile_dim, const int F)
{
    extern __shared__ float aTile[];
    float *bTile = aTile + (tile_dim * tile_dim);

    float sum = 0.0f;

    int row, col;

    for (int tileIdx = 0; tileIdx < K / tile_dim; tileIdx++)
    {
        row = blockIdx.y * blockDim.y + threadIdx.y;
        col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= (M - F))
            break;
        

        aTile[threadIdx.y * tile_dim + threadIdx.x] = a[row * K + tileIdx * tile_dim + threadIdx.x];
        bTile[threadIdx.y * tile_dim + threadIdx.x] = b[threadIdx.y * N + tileIdx * tile_dim * N + col];

        __syncthreads(); // warp usa datos de B leídos por otro warp

        for (int i = 0; i < tile_dim; i++)
        {
            sum += aTile[threadIdx.y * tile_dim + i] * bTile[i * tile_dim + threadIdx.x];
        }

        // __syncthreads(); 
    }

    if (row < (M - F))
        c[row * N + col] = sum;
}