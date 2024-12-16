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

	addDarkPadding(d_imagePtr, width, height, slice, globalThreshold);
	std::cerr << "add darkpadding cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	int* d_compress; //ѹ��ӳ�����
	int* d_decompress; //��ѹ��ӳ�����
	unsigned char* d_imagePtr_compact; //ѹ����ԭͼ
	int newSize; //ѹ���������ܴ�С
	
	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

	std::cerr << "OldSize: " << width * height * slice << " NewSize: " << newSize << std::endl;
	printf("Compress Ratio: %.2lf%%\n", 100.0 * newSize / (1.0 * width * height * slice));

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


	cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize);
	cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);

	

	featureTransForm(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "Feature Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	cudaMemcpy(h_imagePtr, d_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyDeviceToHost);





	int* d_ftarr;
	cudaMalloc(&d_ftarr, sizeof(int) * width * height * slice);
	findFtPoints(d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);
	
	int* h_ftarr = (int*)malloc(sizeof(int) * width * height * slice);
	cudaMemcpy(h_ftarr, d_ftarr, sizeof(int) * width * height * slice, cudaMemcpyDeviceToHost);
	
	unsigned char* h_distPtr = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);
	convertFtPoints2Dist(h_ftarr, h_distPtr, width, height, slice);



	//findFtPointsHost(h_decompress, h_ftarr, h_parentPtr_compact, width, height, slice, newSize);


	//free(h_parentPtr_compact);
	//free(h_decompress);
	


	std::string outputFile = "output.tif";

	std::string reconstructedOutputFile = "rec_output.tif";

	saveTiff(outputFile.c_str(), h_imagePtr, size);

	saveTiff(reconstructedOutputFile.c_str(), h_distPtr, size);

	free(h_distPtr);
	free(h_ftarr);

	free(h_imagePtr);
	cudaFree(d_imagePtr);
	cudaFree(d_imagePtr_compact);
	cudaFree(d_compress);
	cudaFree(d_decompress);
	cudaFree(d_activeMat_compact);
	cudaFree(d_parentPtr_compact);
	cudaFree(d_ftarr);
	return 0;
}
