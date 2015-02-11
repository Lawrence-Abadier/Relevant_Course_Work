// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE BLOCK_SIZE*2

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void postScanArraySum(float * output, float * aux, int len){
    //adding our auxiliary array to the final scanned output
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < len){
        for (int j=0; j < i/blockDim.x; j++)
            output[i] += aux[j];
    }
}

__global__ void scan(float * input, float * output, float * aux, int len) {

    __shared__ float XY[SECTION_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x*blockDim.x;
    
    XY[tid] = (start + tid < len) ? input[start + tid] : 0.0f;
    XY[blockDim.x+tid] = (start + blockDim.x + tid < len) ? input[start + blockDim.x+tid] : 0.0f;

    //Reduction phase
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        //threadID = 1,2,3,4,5 , stride = 1
        //index 1,3,5,7
        int index = (threadIdx.x+1) * 2* stride -1;
        if (index < blockDim.x) {
            //XY[index] = 0, 2, 4, 6
            XY[index] += XY[index -stride];
        }
    }
    //Post Reduction Reverse Phase
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 -1;
        if(index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    
    int i= start + tid;
    //Calculating our output array
    if(i < len)
        output[i] = XY[tid];
        //Our auxiliary array for calculating the final sum of each section
        if ((i+1)% blockDim.x == 0) 
            aux[i/blockDim.x] = output[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * deviceAux;
    int numElements; // number of elements in the list
    
    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The number of input elements in the input is ", numElements);
    
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceAux, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(deviceAux, 0, numElements/BLOCK_SIZE*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");
    
    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(ceil(numElements/SECTION_SIZE));
    dim3 dimBlock(BLOCK_SIZE);
    
    wbTime_start(Compute, "Performing CUDA computation");
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAux, numElements);
    postScanArraySum<<<dimGrid, dimBlock>>>(deviceOutput, deviceAux, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAux);
    wbTime_stop(GPU, "Freeing GPU Memory");
               
    wbSolution(args, hostOutput, numElements);
    
    free(hostInput);
    free(hostOutput);
    
    return 0;
}