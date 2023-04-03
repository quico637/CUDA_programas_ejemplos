


/* Cada thread calcula el elemento (row,col) de C recorriendo la fila row de A y la
columna col de B*/

__global__ void simpleMultiply(float *a, float *b, float *c, int N, const int tile_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < tile_dim; i++)
        sum += a[row * tile_dim + i] * b[i * N + col];

    c[row * N + col] = sum;
}

/* Cada elemento del tile de A se lee de memoria global a memoria compartida solamente una vez, en
forma completamente coalesced, sin desaprovechar ancho de banda */

__global__ void coalescedMultiply(float *a, float *b, float *c, int N, const int tile_dim)
{
    __shared__ float aTile[][];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row * tile_dim + threadIdx.x];

    for (int i = 0; i < tile_dim; i++)
    {
        sum += aTile[threadIdx.y][i] * b[i * N + col];
    }

    c[row * N + col] = sum;
}

/* Para calcular cada fila del tile de C se lee el tile entero de B. Por tanto, para el tile entero de C (el trabajo que hace un bloque de threads) se lee el tile entero de B 
repetidamente (w veces) */

// __global__ void sharedABMultiply(float *a, float *b, float *c, int N, int tile_dim)
// {
//     __shared__ float aTile[tile_dim][tile_dim],

//     bTile[tile_dim][tile_dim];
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0.0f;

//     aTile[threadIdx.y][threadIdx.x] = a[row * tile_dim + threadIdx.x];
//     bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];

//     __syncthreads(); // warp usa datos de B leídos por otro warp

//     for (int i = 0; i < tile_dim; i++)
//     {
//         sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
//     }
//     c[row * N + col] = sum;
// }