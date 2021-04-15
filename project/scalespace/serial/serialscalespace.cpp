#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "../../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb/stb_image_write.h"
#include <sys/time.h>
#include <string>
#define PI 3.14159265358979323846

#define WINDOW 6
#define NUM_ITER 64


int main(int argc, char** argv) {
    	//Initiialize variables for image
    	int width, height, bpp;

    	//initialize timing variables
    	struct timeval start, end;

    	//load in grayscale image
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

    	//initialize image to store result
    	uint8_t* finalimg = new uint8_t[width*height];

    	//initialize scale-space kernel
    	int winsize = (2*WINDOW+1)*(2*WINDOW+1);
    	float* scalekernel = new float[winsize];
	gettimeofday(&start,NULL);
    	//loop over scalepsace iterations
    	for(int iter = 1; iter <= NUM_ITER; ++iter){

    		//Re-define scale-space kernel at each iteration
	        float t = iter*1.0;
        	for (int i = -WINDOW; i <= WINDOW; ++i){
            		for(int j = -WINDOW; j <= WINDOW; ++j){
		                scalekernel[j+WINDOW + (i+WINDOW)*(2*WINDOW+1)] = exp(-(i*i+j*j)/(2*t))/(2*PI*t);
            		}
        	}
    		// Loop over image pixels
	        for(int i = 0; i < width*height; ++i){
        		float sum = 0;
            		for(int p = -WINDOW; p <= WINDOW; ++p){ 	//loop over window
		                for(int q = -WINDOW; q <= WINDOW; ++q){
                			int pixel = i + p*width + q;

                			if((pixel < 0) | (pixel > height*width-1))  //if outside of vertical bound, continue
                    				continue;
                			int temp = i % width;
                			if((temp + q < 0) | (temp + q > width)) //if outside of horiz bound, continue
                    				continue;
               	 			sum += grayimg[pixel]*scalekernel[(p+WINDOW)*(2*WINDOW+1)+(q+WINDOW)];
                		}
            		}
			//store final image
            		finalimg[i] = (uint8_t)((int)sum);
        	}
        	//save images!
        	char filename[64];
        	sprintf (filename, "../../images/Serialscalespace/serial%i.jpg", iter);
        	stbi_write_jpg(filename, width, height, 1, finalimg, 100);
    	}
    	gettimeofday(&end,NULL);
		//print final time
    	printf("Done!\n");
    	std::cout << (double)((end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec)/1000000.0 << '\n';

		//free arrays
    	delete [] grayimg;
    	delete [] finalimg;
    	delete [] scalekernel;

	return 0;
}

