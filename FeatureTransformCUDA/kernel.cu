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
#include "image.h"
#include "volume.h"
#include <algorithm>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

const int GAMMA = 1;
const int FOREGROUND = 1;
const int BACKGROUND = 0;
const int SKEL = 255;

#define sqr(x) ((x)*(x))
#define sqrlength(i, j, k) ((i)*(i)+(j)*(j)+(k)*(k))

static int			gamma_val = GAMMA;
static IntVolume*	ft, *ft_test;
static ByteVolume*	indata, *indata_test;
static char			input_file[MAXSTR];
static char			output_file[MAXSTR];
static char*		basefilename;
static char			basename[MAXSTR];
static char			skel_file[MAXSTR];

void CDT2MAT(BYTE* skel, BYTE* dis, int xdim, int ydim, int zdim)
{
	int counter = 0;
	for (int z = 1; z < zdim; z++) {
		for (int y = 1; y < ydim; y++) {
			for (int x = 1; x < xdim; x++) {
				int left = dis[x - 1 + y * xdim + z * xdim * ydim];
				int top = dis[x + (y - 1) * xdim + z * xdim * ydim];
				int front = dis[x + y * xdim + (z - 1) * xdim * ydim];
				int left_top = dis[x - 1 + (y - 1) * xdim + z * xdim * ydim];
				int left_front = dis[x - 1 + y * xdim + (z - 1) * xdim * ydim];
				int top_front = dis[x + (y - 1) * xdim + (z - 1) * xdim * ydim];
				int now = dis[x + y * xdim + z * xdim * ydim];
				if (max(max(left_top, max(top_front, left_front)), max(left, max(top, front))) > now)
				{
					skel[x + y * xdim + z * xdim * ydim] = SKEL;
					counter++;
				}
					
			}
		}
	}
	std::cerr << "SKEL counter: " << counter << std::endl;
}

__global__ void processArrayKernel(unsigned char* data, int size, int FOREGROUND, int SKEL) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		if (data[idx] == FOREGROUND) {
			data[idx] = 0;
		}
		if (data[idx] == SKEL) {
			data[idx] = 1;
		}
	}
}

void finalProcess(ByteVolume* indata, int allsize) {
	unsigned char* d_indata;

	cudaMalloc(&d_indata, allsize * sizeof(unsigned char));
	cudaMemcpy(d_indata, indata->data, allsize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (allsize + blockSize - 1) / blockSize;

	processArrayKernel << <numBlocks, blockSize >> > (d_indata, allsize, FOREGROUND, SKEL);

	cudaMemcpy(indata->data, d_indata, allsize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(d_indata);
}

void finalProcessRaw(unsigned char* indata_arr, int allsize) {
	unsigned char* d_indata;

	cudaMalloc(&d_indata, allsize * sizeof(unsigned char));
	cudaMemcpy(d_indata, indata_arr, allsize * sizeof(unsigned char), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (allsize + blockSize - 1) / blockSize;

	processArrayKernel << <numBlocks, blockSize >> > (d_indata, allsize, FOREGROUND, SKEL);

	cudaMemcpy(indata_arr, d_indata, allsize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(d_indata);
}

__global__ void modifyArray(unsigned char* data, int size, unsigned char* maxVal, int globalTh) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (data[idx] < globalTh) {
			data[idx] = 0;
		}
		atomicMax((int*)maxVal, (int)data[idx]);
	}
}

__global__ void limitMaxValue(unsigned char* data, int size, int FOREGROUND) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		if (data[idx] > FOREGROUND) {
			data[idx] = FOREGROUND;
		}
	}
}


void dealWithInput(unsigned char* h_inputImagePtr, int allsize, int max) {

	int globalTh = 2;
	unsigned char* d_data;
	unsigned char* d_maxVal;

	cudaMalloc(&d_data, allsize * sizeof(unsigned char));
	cudaMalloc(&d_maxVal, sizeof(unsigned char));

	cudaMemcpy(d_data, h_inputImagePtr, allsize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maxVal, &max, sizeof(unsigned char), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (allsize + blockSize - 1) / blockSize;
	modifyArray << <numBlocks, blockSize >> > (d_data, allsize, d_maxVal, globalTh);

	cudaMemcpy(&max, d_maxVal, sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (max > 1) {
		limitMaxValue << <numBlocks, blockSize >> > (d_data, allsize, FOREGROUND);
	}

	cudaMemcpy(h_inputImagePtr, d_data, allsize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	cudaFree(d_maxVal);
}


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
	int globalThreshold = 3;

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

	unsigned char imageMaxValue;
	//dealWithInput(h_imagePtr, width * height * slice, imageMaxValue);
	std::cerr << "imageMaxValue: " << (int)imageMaxValue << std::endl;
	
	cudaDeviceSynchronize();
	std::cerr << "dealwithInput cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
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
	
	addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
	cudaDeviceSynchronize();
	std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();
	

	//调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	cudaDeviceSynchronize();
	std::cerr << "RecoverImage cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();






	/////////////测试1完毕


	/////////////测试2：在变换距离值的过程中同时存储扩展的父亲信息，直到找到每个点距离最近的背景点坐标存储在ft数组中
	//
	//int* d_parentPtr_compact;
	//unsigned char* d_activeMat_compact;
	//cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);
	//cudaMalloc(&d_activeMat_compact, sizeof(unsigned char) * newSize);
	//cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize);
	//cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);

	//
	////输入：原图d_imagePtr_compact，将原图修改为距离值，并且额外计算d_parentPtr_compact，即扩展的父亲节点信息
	//featureTransForm(d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, width, height, slice, newSize);

	//cudaDeviceSynchronize();
	//std::cerr << "Feature Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();

	////调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	//recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	//cudaDeviceSynchronize();
	//std::cerr << "RecoverImage cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();
	//
	//
	//int* d_ftarr;
	//cudaMalloc(&d_ftarr, sizeof(int) * width * height * slice);

	////ftarr数组即为IMA3D 要求的，距离最近背景点的位置。根据之前计算的parent信息倒着跟踪计算得到
	//findFtPoints(d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);

	//cudaDeviceSynchronize();
	//std::cerr << "findFtPoints cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();

	//int* h_ftarr = (int*)malloc(sizeof(int) * width * height * slice);
	//cudaMemcpy(h_ftarr, d_ftarr, sizeof(int) * width * height * slice, cudaMemcpyDeviceToHost);

	//unsigned char* h_imagePtr_cpy = loadImage(input_file, size);
	//int count = 0;
	//int count_dif = 0;
	//for (int i = 0; i < width * height * slice; ++i) {
	//	if ((int)h_imagePtr_cpy[i] != 0) {
	//		count++;
	//		if (h_ftarr[i] == i)
	//			count_dif++;
	//	}
	//}

	//cout << count << endl;
	//cout << count_dif << endl;

	//////验证：使用ftarr中的位置重新计算距离值，存储在重建数组h_distPtr中
	//unsigned char* h_recDistPtr = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);
	//convertFtPoints2Dist(h_ftarr, h_recDistPtr, width, height, slice);
	//
	////有需要可以调用host版本的findFtPoint
	////findFtPointsHost(h_decompress, h_ftarr, h_parentPtr_compact, width, height, slice, newSize);
	////free(h_parentPtr_compact);
	////free(h_decompress);

	
	////rec_output.tif存储使用ft数组重建得到的距离图
	//std::string reconstructedOutputFile = "rec_output.tif";
	//saveTiff(reconstructedOutputFile.c_str(), h_recDistPtr, size);
	//free(h_recDistPtr);
	//free(h_ftarr);

	//cudaFree(d_activeMat_compact);
	//cudaFree(d_parentPtr_compact);
	//cudaFree(d_ftarr);

	////////测试2完毕


	//output.tif存储featureTransForm()得到的距离图
	std::string outputFile = "output.tif";

	unsigned char* h_distPtr = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);
	cudaMemcpy(h_distPtr, d_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyDeviceToHost);

	saveTiff("tempimage.tif", h_distPtr, size);


	cudaDeviceSynchronize();
	std::cerr << "Malloc cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	for (int i = 0; i < width * height * slice; i++)
	{
		if (h_imagePtr[i] < globalThreshold) h_imagePtr[i] = 0;
		else h_imagePtr[i] = 1;
	}
	
	saveTiff("tempimage2.tif", h_imagePtr, size);


	CDT2MAT(h_imagePtr, h_distPtr, xdim, ydim, zdim);

	cudaDeviceSynchronize();
	std::cerr << "CDT2MAT cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	finalProcessRaw(h_imagePtr, width * height * slice);

	cudaDeviceSynchronize();
	std::cerr << "FinalProcess cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	saveTiff(outputFile.c_str(), h_imagePtr, size);

	free(h_imagePtr);
	free(h_distPtr);
	cudaFree(d_imagePtr);
	cudaFree(d_imagePtr_compact);
	cudaFree(d_compress);
	cudaFree(d_decompress);
	
	return 0;
}
