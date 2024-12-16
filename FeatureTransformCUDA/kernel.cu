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

	std::cerr << "Usage: inputname threshold" << std::endl;

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

	
	//添加全局阈值
	addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);

	std::cerr << "set globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//将原图中紧靠前景点的背景点赋值为1，作为之后的扩展起点
	addDarkPadding(d_imagePtr, width, height, slice, globalThreshold);
	std::cerr << "add darkpadding cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	int* d_compress; //压缩映射矩阵
	int* d_decompress; //解压缩映射矩阵
	unsigned char* d_imagePtr_compact; //压缩后原图
	int newSize; //压缩后数组总大小
	
	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

	std::cerr << "OldSize: " << width * height * slice << " NewSize: " << newSize << std::endl;
	printf("Compress Ratio: %.2lf%%\n", 100.0 * newSize / (1.0 * width * height * slice));

	cudaDeviceSynchronize();
	std::cerr << "compaction cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	////////////////////以上为输入


	
	//测试1: 直接将原图变换为距离值，调用addGreyWeightTransform() 函数 将d_imagePtr_compact 变为距离变换之后的数组
	
	//addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
	//cudaDeviceSynchronize();
	//std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();

	//调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	//recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);



	/////////////测试1完毕


	///////////测试2：在变换距离值的过程中同时存储扩展的父亲信息，直到找到每个点距离最近的背景点坐标存储在ft数组中
	
	int* d_parentPtr_compact;
	unsigned char* d_activeMat_compact;
	cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);
	cudaMalloc(&d_activeMat_compact, sizeof(unsigned char) * newSize);
	cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize);
	cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);

	
	//输入：原图d_imagePtr_compact，将原图修改为距离值，并且额外计算d_parentPtr_compact，即扩展的父亲节点信息
	featureTransForm(d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "Feature Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	cudaDeviceSynchronize();
	std::cerr << "RecoverImage cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();
	
	
	int* d_ftarr;
	cudaMalloc(&d_ftarr, sizeof(int) * width * height * slice);

	//ftarr数组即为IMA3D 要求的，距离最近背景点的位置。根据之前计算的parent信息倒着跟踪计算得到
	findFtPoints(d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "findFtPoints cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	int* h_ftarr = (int*)malloc(sizeof(int) * width * height * slice);
	cudaMemcpy(h_ftarr, d_ftarr, sizeof(int) * width * height * slice, cudaMemcpyDeviceToHost);


	////验证：使用ftarr中的位置重新计算距离值，存储在重建数组h_distPtr中
	unsigned char* h_recDistPtr = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);
	convertFtPoints2Dist(h_ftarr, h_recDistPtr, width, height, slice);
	
	//有需要可以调用host版本的findFtPoint
	//findFtPointsHost(h_decompress, h_ftarr, h_parentPtr_compact, width, height, slice, newSize);
	//free(h_parentPtr_compact);
	//free(h_decompress);
	
	//output.tif存储featureTransForm()得到的距离图
	std::string outputFile = "output.tif";

	//rec_output.tif存储使用ft数组重建得到的距离图
	std::string reconstructedOutputFile = "rec_output.tif";

	cudaMemcpy(h_imagePtr, d_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyDeviceToHost);

	saveTiff(outputFile.c_str(), h_imagePtr, size);

	saveTiff(reconstructedOutputFile.c_str(), h_recDistPtr, size);

	////////测试2完毕

	free(h_recDistPtr);
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
