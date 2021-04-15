#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "../../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb/stb_image_write.h"
#include <algorithm>
#include "mpi.h"

#include <sys/time.h>

#define PI 3.14159265358979323846

#define WINDOW 6
#define NUM_ITER 64

uint8_t* launch_cuda(float *grayimg, float *scalekernel, uint8_t *finalimg, int height, int width, int window);

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

	//initialize image to store result, initialize variable for device, and allocate space in device
	uint8_t *finalimg = new uint8_t[width*height];

    	//Initialize and define scale-space kernel
	float t;
     	int winsize = (2*WINDOW+1)*(2*WINDOW+1);
     	float* scalekernel = new float[winsize];

	//initialize MPI
	int mpi_size, mpi_rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_num_iter = ceil(NUM_ITER/(float)mpi_size);
	
	//start loop for scalespace iterations
	start = MPI_Wtime();
	for(int iter = 1 + mpi_rank*mpi_num_iter; iter < std::min(1 + NUM_ITER,1 + mpi_num_iter*(mpi_rank + 1)); ++iter){
		t = iter*1.0;

		//update scale-space kernel
	        for (int i = -WINDOW; i <= WINDOW; ++i){
	        	for(int j = -WINDOW; j <= WINDOW; ++j){
                		scalekernel[j+WINDOW + (i+WINDOW)*(2*WINDOW+1)] = exp(-(i*i+j*j)/(2*t))/(2*PI*t);
            		}
        	 }

	        
        	// launch cuda kernels using CUDA function
	        launch_cuda(grayimg, scalekernel, finalimg, height, width, WINDOW);

		//save image!
    		char filename[64];
		sprintf (filename, "../../images/Mpicudascalespace/mpicuda%i.jpg", iter);
		stbi_write_jpg(filename, width, height, 1, finalimg, 100);
	}
	end = MPI_Wtime();
	if(mpi_rank == 0) 
		std::cout << end-start << '\n';

	//free host variables
    	delete [] grayimg;
		delete [] scalekernel;
    	delete [] finalimg;
	
	//finalize MPI
	MPI_Finalize();
	return 0;
}

