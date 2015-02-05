#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)
#define clamp(x, start, end) (min(max((x), 0.0), 1.0))

__global__ void convolution_2D_kernel(float* OutputImageData, float* InputImageData,
int Height, int Width, int Channels,
const float* __restrict__ MaskData){
    //Our tiled input image data that is
    //shared amongst our threads
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH][3];
    
    //setting up our thread ID's
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //Our output image value
    float outputValue = 0.0f;
    
    //Our OutputImageData index
    int row_out = blockIdx.y * O_TILE_WIDTH + ty;
    int col_out = blockIdx.x * O_TILE_WIDTH + tx;
    
    //our tiled input index
    int row_in = row_out - Mask_radius;
    int col_in = col_out - Mask_radius;
    
    for(int k = 0; k < Channels; k++){
        //Loading input tiles in our shared array
        if((row_in >=0 && row_in < Height)&&(col_in >= 0 && col_in < Width)){
            Ns[ty][tx][k] = InputImageData[(row_in * Width + col_in)*Channels + k];
        }
        else{
            Ns[ty][tx][k] = 0.0f;
        }
        __syncthreads();
		
        //Only threads 0 to Tile width participate in calculating OutputImageData
        if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
            outputValue = 0.0f;
            for(int i = 0; i < Mask_width; i++){
                for(int j = 0; j < Mask_width; j++){
                    outputValue += MaskData[i * Mask_width + j] * Ns[i + ty][j+tx][k];
                }
            }
        }
		__syncthreads();
        //threads outside of our bounds do not write to output
        if(row_out < Height && col_out < Width && tx < O_TILE_WIDTH && ty < O_TILE_WIDTH)
        	OutputImageData[(row_out * Width + col_out) * Channels + k] = clamp(outputValue, 0, 1);
        	
    }
}  


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 DimGrid((imageWidth -1)/ O_TILE_WIDTH + 1, (imageHeight -1)/ O_TILE_WIDTH + 1);
	dim3 DimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);
		
	
	convolution_2D_kernel<<<DimGrid, DimBlock>>>(deviceOutputImageData, deviceInputImageData, 
    	imageHeight, imageWidth, imageChannels,
    	deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
