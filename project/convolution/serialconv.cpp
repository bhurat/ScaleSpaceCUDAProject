#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"
#include <sys/time.h>

#define PI 3.14159265358979323846
#define WINDOW 6

int main(int argc, char** argv) {
    //Initiialize variables for image
    int width, height, bpp;

	//initialize timing variables
	struct timeval start, end;

	//load in color image
	uint8_t* img = stbi_load("../images/imageBig.jpg", &width, &height, &bpp, 3);
	if(img == NULL) {
         	 printf("Error in loading the image\n");
	         exit(1);
    }
    //convert to grayscale and free image
    float* grayimg = new float[width*height];
    for(int i = 0; i < width*height; ++i)
        grayimg[i] = (.3*(int)img[3*i] + .59*(int)img[3*i+1]+.11*(int)img[3*i+2]);
    stbi_image_free(img);

    //Initialize and define scale-space kernel
     float t = 1.0;
     int winsize = (2*WINDOW+1)*(2*WINDOW+1);
     float* scalekernel = new float[winsize];
     for (int i = -WINDOW; i <= WINDOW; ++i){
        for(int j = -WINDOW; j <= WINDOW; ++j){
            scalekernel[j+WINDOW + (i+WINDOW)*(2*WINDOW+1)] = exp(-(i*i+j*j)/(2*t))/(2*PI*t);	//2D sampled gaussian kernel
        }
     }

    //initialize image to store result
    uint8_t* finalimg = new uint8_t[width*height];

	//start clock and loop over image pixels
    gettimeofday(&start,NULL);
    
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
    gettimeofday(&end,NULL);
	//print time
    printf("Done!\n");
    std::cout << (double)((end.tv_sec-start.tv_sec)*1000000 + end.tv_usec-start.tv_usec)/1000000.0 << '\n';
	//save image
	int channels = 1;
    stbi_write_jpg("../images/convserial.jpg", width, height, channels, finalimg, 100);
    
	//free memory
	delete [] grayimg;
    delete [] finalimg;
    delete [] scalekernel;
	
	return 0;
}

