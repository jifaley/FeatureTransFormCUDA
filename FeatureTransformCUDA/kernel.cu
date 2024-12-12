#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "loadTiff.h"
#include "TimerClock.hpp"
#include "compaction.h"
#include "fastmarching.h"
#include "threshold.h"

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>


int main(int argc, const char **argv)
{
	int i;
	int infile = -1;
	int outfile = -1;
	int gammavalue = -1;
	int xdim, ydim, zdim;

	TimerClock timer;

	std::cerr << "Begin " << std::endl << std::endl;
	timer.update();

	std::string input_file;
	int globalThreshold = 5;

	if (argc > 1)
		input_file = argv[1];

	if (argc > 2)
	{
		if (atoi(argv[2]) != -1)
			globalThreshold = atoi(argv[2]);
	}

	auto size = new int[3];
	unsigned char* h_imagePtr = loadImage(input_file, size);
	int width = size[0];
	int height = size[1];
	int slice = size[2];

	unsigned char* d_imagePtr = nullptr;
	cudaMalloc(&d_imagePtr, sizeof(unsigned char) * width * height * slice);
	cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyHostToDevice);


	std::cerr << "Load cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	

	addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);

	std::cerr << "set globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();



	int* d_compress; //Ñ¹ËõÓ³Éä¾ØÕó
	int* d_decompress; //½âÑ¹ËõÓ³Éä¾ØÕó
	unsigned char* d_imagePtr_compact; //Ñ¹ËõºóÔ­Í¼
	int newSize; //Ñ¹ËõºóÊý×é×Ü´óÐ¡
	
	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

	std::cerr << "OldSize: " << width * height * slice << " NewSize: " << newSize << std::endl;
	printf("Compress Ratio: %.2lf%%\n", newSize *100/ (1.0 * width * height * slice));

	cudaDeviceSynchronize();
	std::cerr << "compaction cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	//addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
	cudaDeviceSynchronize();
	std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	
	int* d_parentPtr_compact;
	cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);


	unsigned char* d_activeMat_compact;
	cudaMalloc(&d_activeMat_compact, sizeof(unsigned char) * newSize);


	cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize * 2);
	cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);

	featureTransForm(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "Feature Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();



	std::string outputFile = "output.tif";

	saveTiff(outputFile.c_str(), h_imagePtr, size);

	free(h_imagePtr);
	cudaFree(d_imagePtr);
	cudaFree(d_imagePtr_compact);
	cudaFree(d_compress);
	cudaFree(d_decompress);
	cudaFree(d_activeMat_compact);
	cudaFree(d_parentPtr_compact);

	return 0;
}
