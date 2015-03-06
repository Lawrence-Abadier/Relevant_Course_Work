// MP7 Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
  wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
  wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
  return -1;                                                        \
  }                                                                     \
    } while(0)

//Cast the float img to unsigned char
__global__ void floatImgToChar(float * inputImage, unsigned char * outputUCharImage, unsigned int outputLen){
  //The absolute thread id
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int threadCount = blockDim.x*gridDim.x;

  for(unsigned int i = 0;i < (outputLen - 1  ) / threadCount + 1; ++i){
    unsigned int indexi = index + i * threadCount;
    if(indexi < outputLen) 
      outputUCharImage[indexi] = (unsigned char) (255 * inputImage[indexi]); 
  }
}

__global__ void rgbToGS(unsigned char * ucharInputImg, unsigned char * grayImage, unsigned int outputLen){
  //Absolute thread id
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  //Total number of threads
  unsigned int threadCount = blockDim.x*gridDim.x;

  unsigned int channels = 3;
  unsigned char r,g,b;

  //OutputImageData[(row_out * Width + col_out) * Channels + k]
  for(int i = 0;i < (outputLen - 1)/threadCount +1;++i){
    unsigned int idx = index + i * threadCount;
    if(idx < outputLen){
      r = ucharInputImg[channels * idx];
      g = ucharInputImg[channels * idx + 1];
      b = ucharInputImg[channels * idx + 2];
      // color coefficients for computing grayscale
      grayImage[idx] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
    }
  }
}

//Commputes the histogram of the img data
__global__ void histogram(unsigned int *histo, unsigned char *grayImage, unsigned int inputLength){

  //Atomics in shared memory require privatization
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  //The local and absolute thread ID
  int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index < HISTOGRAM_LENGTH) 
    histo[index] = 0;   
  //Initialize the histo_private array to 0
  if(threadIdx.x < HISTOGRAM_LENGTH) 
    histo_private[tid] = 0; 
  __syncthreads();

  //The total number of threads
  unsigned int threadCount = blockDim.x*gridDim.x;

  //Compute atomic adds to shared memory
  for(int i = 0;i < (inputLength - 1)/threadCount + 1;++i){
    //Each thread works on multiple pixels based on image size
    unsigned int indexi = index + i*threadCount;
    if(indexi < inputLength){
      atomicAdd( & histo_private[grayImage[indexi]], 1);
    }
  }
  //Wait for all the other threads in the block to finish
  __syncthreads();

  //We add the values to our histo array from shared memory
  if(tid < HISTOGRAM_LENGTH) 
    atomicAdd( & histo[tid], 
    histo_private[tid]);
}

//This is a prefix-sum of all the luminosity values in our image
__global__ void cumulativeDistribution(unsigned int *histo, float *histoCumulative, int widthByHeight){

  histoCumulative[0] = histo[0]/(float) widthByHeight;

  for(int i = 1; i < HISTOGRAM_LENGTH; i++){
    __syncthreads();
    histoCumulative[i] = histoCumulative[i-1] + (histo[i]/(float) widthByHeight);
  }
}

//Clamps min and max values
__device__ int clamp(int x, int start, int end){
  return min(max(x, start), end);
}

//Remaps the cdf of the histogram of the image to a linear function
__device__ int colorCorrect(int val, float* cdf){
  return clamp(255*(cdf[val] - cdf[0]/ (1 - cdf[0])), 0, 255);
}

//We correct the input image's color and cast the image data back to floats
__global__ void postEqualization(float* outputImage, float * cdf, unsigned char *ucharInputImg, unsigned int outputLen){

  //The absolute thread id
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int threadCount = blockDim.x*gridDim.x;

  for(unsigned int i = 0;i < (outputLen - 1  ) / threadCount + 1; ++i){

    unsigned int indexi = index + i * threadCount;
    //If our threads are within bounds we cast the
    //cumulative distribution data to floats with corrected colors
    if(indexi < outputLen)
      outputImage[indexi] = (float)(colorCorrect(ucharInputImg[indexi], cdf)/255.0);
  }
}


int main(int argc, char ** argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * hostInputImageData;
  float * hostOutputImageData;
  const char * inputImageFile;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);


  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage); 

  // initialize foo
  unsigned int pixelCount = imageWidth * imageHeight;
  unsigned int channelCount = pixelCount * imageChannels;
  // compute bar
  float * deviceInputImageData;
  wbCheck(cudaMalloc((void**) & deviceInputImageData, channelCount * sizeof(float)));
  wbCheck(cudaMemcpy(deviceInputImageData,hostInputImageData,channelCount*sizeof(float),cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  dim3 BlockSize = BLOCK_SIZE;
  dim3 GridSizeChannels = (channelCount > (BlockSize.x<<2)) ? 4 : (channelCount-1)/BlockSize.x + 1;
  dim3 GridSizePixels = (pixelCount > (BlockSize.x<<2)) ? 4 : (pixelCount-1)/BlockSize.x + 1;

  unsigned char * deviceUcharImg;
  wbCheck(cudaMalloc((void**) & deviceUcharImg, channelCount * sizeof(unsigned char)));
  floatImgToChar<<<GridSizeChannels, BlockSize>>>(deviceInputImageData, deviceUcharImg, channelCount);

  cudaDeviceSynchronize();

  unsigned char * grayImage;
  wbCheck(cudaMalloc((void**) & grayImage, pixelCount * sizeof(unsigned char)));
  rgbToGS<<<GridSizePixels, BlockSize>>>(deviceUcharImg, grayImage, pixelCount);

  cudaDeviceSynchronize();

  unsigned int * devicehisto;
  wbCheck(cudaMalloc((void**) & devicehisto, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  histogram<<<GridSizePixels, BlockSize>>>(devicehisto,grayImage, pixelCount); 

  cudaDeviceSynchronize();

  float * devicehistogramCDF;
  wbCheck(cudaMalloc((void**) & devicehistogramCDF, HISTOGRAM_LENGTH*sizeof(float)));
  cumulativeDistribution<<<1,HISTOGRAM_LENGTH/2>>>(devicehisto, devicehistogramCDF, pixelCount);
  cudaDeviceSynchronize();

  float * deviceOutputImage;
  wbCheck(cudaMalloc((void**) & deviceOutputImage, channelCount * sizeof(float)));
  postEqualization<<<GridSizeChannels,BlockSize>>>(deviceOutputImage, devicehistogramCDF, deviceUcharImg, channelCount); 

  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutputImageData,deviceOutputImage,channelCount*sizeof(float),cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
  wbSolution(args, outputImage);
  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceUcharImg);
  cudaFree(grayImage);
  cudaFree(devicehisto);
  cudaFree(devicehistogramCDF);
  cudaFree(deviceOutputImage);

  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}