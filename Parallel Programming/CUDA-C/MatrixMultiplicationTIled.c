#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, 
                int numAColumns, int numBColumns, int numCColumns) {
  //@@ Insert code to implement tiled matrix multiplication here
  //Creating out shared tile
  __shared__ float A_d[TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_d[TILE_WIDTH][TILE_WIDTH];

  //setting up thread and block id
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x;int ty = threadIdx.y;
  //Identifying the correct C[row][col] to work on
  //Row/Collumn major layout
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  
  for ( int i = 0; i < (numAColumns-1)/TILE_WIDTH+1 ; ++i){
    //initializing our C output value back to 0
    float Cvalue = 0.0;  
  
    if (Row < numARows  && (i * TILE_WIDTH + tx) < numAColumns) {      
      //loading the tiled section of A into shared memory
      A_d[ty][tx] = A[Row * numAColumns + (i * TILE_WIDTH + tx)];
      __syncthreads();
    }
    else{
      //if the tile is out of matrix A's' bounds, initilize the index to 0
      A_d[ty][tx] = 0;
    }
    if (Col < numBColumns  && (i * TILE_WIDTH + ty) < numAColumns) {      
      //loading the tiled section of B into shared memory
      B_d[ty][tx] = B[Col + numBColumns * (i *TILE_WIDTH + ty)];
      __syncthreads();
    }
    else{
      //if the tile is out of matrix B's bounds, initilize the index to 0
      B_d[ty][tx] = 0;
    }

    for(int j = 0; j < TILE_WIDTH; j++){
      Cvalue += A_d[ty][j] * B_d[j][ty];
    }
    __syncthreads();

    //We only set Cvalues that are within our C[ROW][COL]
    //Preventing any previous 0 values to be included
    if(Row < numARows && Col < numBColumns){
      C[Row * numCColumns + Col] = Cvalue;
    }
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = ( float * )malloc(((numARows) * (numBColumns )) * sizeof(float)); 
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  
  //Initilizing the size in bytes
  int sizeA = (numARows*numAColumns)*sizeof(float);
  int sizeB = (numBRows*numBColumns)*sizeof(float);
  int sizeC = (numCRows*numCColumns)*sizeof(float);

  //Checking for errors
  cudaError_t errA = cudaMalloc((void **) &deviceA, sizeA);
  cudaError_t errB = cudaMalloc((void **) &deviceB, sizeB);
  cudaError_t errC = cudaMalloc((void **) &deviceC, sizeC);
  
  if((errA != cudaSuccess)&&(errB != cudaSuccess)&&(errC != cudaSuccess)){
    //Exit on failure
    exit(EXIT_FAILURE);
  }else{
    //Allocate device memory
    cudaMalloc((void **) &deviceA, sizeA);
    cudaMalloc((void **) &deviceB, sizeB);
    cudaMalloc((void **) &deviceC, sizeC);
    //Copy host memory to device
    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, hostC, sizeC, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU."); 
  }

  
  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
  dim3 gridDim(ceil(numCColumns-1/blockDim.x), ceil(numCRows-1/blockDim.y));
 
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<gridDim,blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  
  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
