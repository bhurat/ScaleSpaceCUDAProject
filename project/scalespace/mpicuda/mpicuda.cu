#include <cuda.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define PI 3.14159265358979323846

#define WINDOW 6
#define NUM_ITER 64

__global__
void conv2(float *A, float *B,uint8_t *C, int height, int width, int window){
        int row = threadIdx.x + blockIdx.x*blockDim.x;
        int col = threadIdx.y + blockIdx.y*blockDim.y;
        int i = row + col*width;
        float sum = 0;
        if(i<height*width){
                for(int p = -window; p <= window; ++p){     //loop over window
                        for(int q = -window; q <= window; ++q){
                                int pixel = i + p*width + q;
                                if((pixel < 0) | (pixel >= height*width))  //if outside of vertical bound, continue
                                        continue;
                                int temp = i % width;
                                if((temp + q < 0) | (temp + q >= width)) //if outside of horiz bound, continue
                                        continue;
                                sum += A[pixel]*B[(p+window)*(2*window+1)+(q+window)];
                        }
                }
                C[i] = (uint8_t)((int)sum);
        }
}

extern uint8_t* launch_cuda(float *grayimg, float *scalekernel, uint8_t *finalimg, int height, int width, int window){
	//initialize and allocate device variables
	int winsize = (2*window + 1)*(2*window + 1);
	float *d_gray, *d_scalekernel;
	uint8_t *d_finalimg;

	cudaMalloc((void**)&d_gray,height*width*sizeof(float));
	cudaMalloc((void**)&d_scalekernel,winsize*sizeof(float));
	cudaMalloc((void**)&d_finalimg,width*height*sizeof(uint8_t));
	
	//copy info to device
	cudaMemcpy(d_gray,grayimg,height*width*sizeof(float),cudaMemcpyHostToDevice); 
	cudaMemcpy(d_scalekernel,scalekernel,winsize*sizeof(float),cudaMemcpyHostToDevice);

        //initialize blocksize and gridsize
        dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
        dim3 dimGrid(ceil(width/(float)dimBlock.x),ceil(height/(float)dimBlock.y));

	// Loop over image pixels by calling CUDA kernel
        conv2<<<dimGrid,dimBlock>>>(d_gray, d_scalekernel, d_finalimg, height, width, window);

        //copy result to host
        cudaMemcpy(finalimg,d_finalimg,height*width*sizeof(uint8_t),cudaMemcpyDeviceToHost);
	
	//free cuda variables
	cudaFree(d_gray);
	cudaFree(d_scalekernel);
	cudaFree(d_finalimg);
	return finalimg;
}
