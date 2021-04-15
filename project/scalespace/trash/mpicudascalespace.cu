#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "../../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb/stb_image_write.h"
#include "mpi.h"
#include <cuda.h>

#include <sys/time.h>

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

int main(int argc, char** argv) {
    	//Initiialize variables for image
    	int width, height, bpp;
	//initialize timing variables for mpi timing
	double start, end;

    	//load in image
	uint8_t* img = stbi_load("../../images/cat.jpg", &width, &height, &bpp, 3);
    	if(img == NULL) {
          	printf("Error in loading the image\n");
         	exit(1);
    	}
    	//convert to grayscale and free image
    	float* grayimg = new float[width*height];
    	for(int i = 0; i < width*height; ++i)
        	grayimg[i] = (.3*(int)img[3*i] + .59*(int)img[3*i+1]+.11*(int)img[3*i+2]);
    	stbi_image_free(img);

	//initialize device copy of grayscale image, allocate it, and copy to device
	float *d_gray;
	cudaMalloc((void**)&d_gray,height*width*sizeof(float));
	cudaMemcpy(d_gray,grayimg,height*width*sizeof(float),cudaMemcpyHostToDevice);


	//initialize image to store result, initialize variable for device, and allocate space in device
	uint8_t* finalimg = new uint8_t[width*height];
	uint8_t* d_finalimg;
	cudaMalloc((void**)&d_finalimg,width*height*sizeof(uint8_t));

    	//Initialize and define scale-space kernel
	float t;
     	int winsize = (2*WINDOW+1)*(2*WINDOW+1);
     	float* scalekernel = new float[winsize];



	//initialize device copy of scale-space kernel, allocate space, and copy to device
	float *d_scalekernel;
	cudaMalloc((void**)&d_scalekernel,winsize*sizeof(float));

	//start timer and loop through scalespace iterations
	int mpi_size, mpi_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_num_iter = ceil(NUM_ITER/(float)mpi_size);
	start = MPI_Wtime();
	for(int iter = 1 + mpi_rank*mpi_num_iter; iter < min(1 + NUM_ITER,mpi_num_iter*(mpi_rank + 1)); ++iter){
		t = iter*1.0;

		//update scale-space kernel
	        for (int i = -WINDOW; i <= WINDOW; ++i){
	        	for(int j = -WINDOW; j <= WINDOW; ++j){
                		scalekernel[j+WINDOW + (i+WINDOW)*(2*WINDOW+1)] = exp(-(i*i+j*j)/(2*t))/(2*PI*t);
            		}
        	 }
		//copy scale-space kernel onto device
        	cudaMemcpy(d_scalekernel,scalekernel,winsize*sizeof(float),cudaMemcpyHostToDevice);

	        //initialize blocksize and gridsize
        	dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	        dim3 dimGrid(ceil(width/(float)dimBlock.x),ceil(height/(float)dimBlock.y));

        	// Loop over image pixels by calling CUDA kernel
	        conv2<<<dimGrid,dimBlock>>>(d_gray, d_scalekernel, d_finalimg, height, width, WINDOW);

	        //copy result to host
        	cudaMemcpy(finalimg,d_finalimg,height*width*sizeof(uint8_t),cudaMemcpyDeviceToHost);

		//save image!
    		char filename[64];
		sprintf (filename, "../../images/Mpicudascalespace/mpicuda%i.jpg", iter);
		stbi_write_jpg(filename, width, height, 1, finalimg, 100);
	}
	end = MPI_Wtime();
	std::cout << end-start << '\n';

	//free device variables
	cudaFree(d_gray);
	cudaFree(d_scalekernel);
	cudaFree(d_finalimg);

	//free host variables
    	delete [] grayimg;
	delete [] scalekernel;
    	delete [] finalimg;
	
	MPI_Finalize();
	return 0;
}

