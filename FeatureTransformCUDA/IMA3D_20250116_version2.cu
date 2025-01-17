#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "volume.h"
#include "loadTiff.h"
#include "TimerClock.hpp"
#include "compaction.h"
#include "fastmarching.h"
#include "threshold.h"

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

const int GAMMA = 1;
const int FOREGROUND = 1;
const int BACKGROUND = 0;
const int SKEL = 255;

#define sqr(x) ((x)*(x))
#define sqrlength(i, j, k) ((i)*(i)+(j)*(j)+(k)*(k))

static int			gamma_val = GAMMA;
static IntVolume*	ft,*ft_test;
static ByteVolume*	indata, *indata_test;
static char			input_file[MAXSTR];
static char			output_file[MAXSTR];
static char*		basefilename;
static char			basename[MAXSTR];
static char			skel_file[MAXSTR];

__global__ void skeletonKernel(BYTE* skel, BYTE* dis, int xdim, int ydim, int zdim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x <= 0 || x >= xdim - 1 || y <= 0 || y >= ydim - 1 || z <= 0 || z >= zdim - 1)
		return;

	int idx = x + y * xdim + z * xdim * ydim;

	if (dis[idx] == 0)
		return;

	int neighborOffsets[26][3] = {
		{-1, -1, -1}, { 0, -1, -1}, { 1, -1, -1},
		{-1,  0, -1}, { 0,  0, -1}, { 1,  0, -1},
		{-1,  1, -1}, { 0,  1, -1}, { 1,  1, -1},

		{-1, -1,  0}, { 0, -1,  0}, { 1, -1,  0},
		{-1,  0,  0},               { 1,  0,  0},
		{-1,  1,  0}, { 0,  1,  0}, { 1,  1,  0},

		{-1, -1,  1}, { 0, -1,  1}, { 1, -1,  1},
		{-1,  0,  1}, { 0,  0,  1}, { 1,  0,  1},
		{-1,  1,  1}, { 0,  1,  1}, { 1,  1,  1},
	};

	BYTE maxNeighborValue = 0;
	for (int i = 0; i < 26; i++) {
		int nx = x + neighborOffsets[i][0];
		int ny = y + neighborOffsets[i][1];
		int nz = z + neighborOffsets[i][2];

		if (nx >= 0 && nx < xdim && ny >= 0 && ny < ydim && nz >= 0 && nz < zdim) {
			if (nx <= x && ny <= y && nz <= z) {
				int neighborIdx = nx + ny * xdim + nz * xdim * ydim;
				maxNeighborValue = fmaxf(maxNeighborValue, dis[neighborIdx]);
			}
		}
	}

	if (dis[idx] < maxNeighborValue) {
		skel[idx] = SKEL;
	}
}

__global__ void markSimpleVoxels(BYTE* dis, int xdim, int ydim, int zdim) {
	// 当前线程的 3D 坐标
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	// 检查是否在图像范围内
	if (x <= 0 || x >= xdim - 1 || y <= 0 || y >= ydim - 1 || z <= 0 || z >= zdim - 1)
		return;

	// 当前体素的线性索引
	int idx = x + y * xdim + z * xdim * ydim;

	// 如果当前体素是背景，直接退出
	if (dis[idx] == 0)
		return;

	// 邻域偏移
	int neighborOffsets[26][3] = {
		{-1, -1, -1}, { 0, -1, -1}, { 1, -1, -1},
		{-1,  0, -1}, { 0,  0, -1}, { 1,  0, -1},
		{-1,  1, -1}, { 0,  1, -1}, { 1,  1, -1},

		{-1, -1,  0}, { 0, -1,  0}, { 1, -1,  0},
		{-1,  0,  0},               { 1,  0,  0},
		{-1,  1,  0}, { 0,  1,  0}, { 1,  1,  0},

		{-1, -1,  1}, { 0, -1,  1}, { 1, -1,  1},
		{-1,  0,  1}, { 0,  0,  1}, { 1,  0,  1},
		{-1,  1,  1}, { 0,  1,  1}, { 1,  1,  1},
	};

	// 记录邻域访问情况
	bool visited[26] = { false };
	int connectedComponents = 0;

	// 遍历 26 邻域进行连通性检查
	for (int i = 0; i < 26; i++) {
		int nx = x + neighborOffsets[i][0];
		int ny = y + neighborOffsets[i][1];
		int nz = z + neighborOffsets[i][2];

		if (nx >= 0 && nx < xdim && ny >= 0 && ny < ydim && nz >= 0 && nz < zdim) {
			int neighborIdx = nx + ny * xdim + nz * xdim * ydim;

			if (dis[neighborIdx] > 0 && !visited[i]) {
				connectedComponents++;

				for (int j = 0; j < 26; j++) {
					int nnx = x + neighborOffsets[j][0];
					int nny = y + neighborOffsets[j][1];
					int nnz = z + neighborOffsets[j][2];

					if (nnx >= nx - 1 && nnx <= nx + 1 &&
						nny >= ny - 1 && nny <= ny + 1 &&
						nnz >= nz - 1 && nnz <= nz + 1) {
						int nNeighborIdx = nnz * xdim * ydim + nny * xdim + nnx;
						if (dis[nNeighborIdx] > 0)
							visited[j] = true;
					}
				}
			}
		}
	}
	if (connectedComponents == 1) {
		dis[idx] = 0;
		return;
	}
}

__global__ void markSkeletonVoxels(BYTE* skel, int xdim, int ydim, int zdim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x <= 0 || x >= xdim - 1 || y <= 0 || y >= ydim - 1 || z <= 0 || z >= zdim - 1)
		return;

	int idx = x + y * xdim + z * xdim * ydim;

	if (skel[idx] != 0) {
		skel[idx] = 255; // 将未被删除的体素标记为骨架
	}
}



__global__ void AddNeighbor(BYTE* skel, int xdim, int ydim, int zdim) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x <= 0 || x >= xdim - 1 || y <= 0 || y >= ydim - 1 || z <= 0 || z >= zdim - 1)
		return;

	int idx = x + y * xdim + z * xdim * ydim;

	int neighborOffsets[26][3] = {
		{-1, -1, -1}, { 0, -1, -1}, { 1, -1, -1},
		{-1,  0, -1}, { 0,  0, -1}, { 1,  0, -1},
		{-1,  1, -1}, { 0,  1, -1}, { 1,  1, -1},

		{-1, -1,  0}, { 0, -1,  0}, { 1, -1,  0},
		{-1,  0,  0},               { 1,  0,  0},
		{-1,  1,  0}, { 0,  1,  0}, { 1,  1,  0},

		{-1, -1,  1}, { 0, -1,  1}, { 1, -1,  1},
		{-1,  0,  1}, { 0,  0,  1}, { 1,  0,  1},
		{-1,  1,  1}, { 0,  1,  1}, { 1,  1,  1},
	};

	for (int i = 0; i < 26; i++) {
		int nx = x + neighborOffsets[i][0];
		int ny = y + neighborOffsets[i][1];
		int nz = z + neighborOffsets[i][2];

		if (nx >= 0 && nx < xdim && ny >= 0 && ny < ydim && nz >= 0 && nz < zdim) {

			int neighborIdx = nx + ny * xdim + nz * xdim * ydim;
			if (skel[neighborIdx] == SKEL)
				skel[idx] = SKEL;

		}
	}
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

__global__ void meanBlurKernel(BYTE* input, BYTE* output, int xdim, int ydim, int zdim, int kernelRadius) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < kernelRadius || x >= xdim - kernelRadius ||
		y < kernelRadius || y >= ydim - kernelRadius ||
		z < kernelRadius || z >= zdim - kernelRadius) {
		return;
	}

	int idx = x + y * xdim + z * xdim * ydim;

	float sum = 0.0f;
	int count = 0;

	for (int dz = -kernelRadius; dz <= kernelRadius; dz++) {
		for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
			for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
				int nx = x + dx;
				int ny = y + dy;
				int nz = z + dz;

				int neighborIdx = nx + ny * xdim + nz * xdim * ydim;

				sum += input[neighborIdx];
				count++;
			}
		}
	}

	output[idx] = (BYTE)(sum / count);
}

void GetSkeleton(unsigned char* ImagePtr, int xdim, int ydim, int zdim)
{
	TimerClock timer;

	unsigned char* d_imagePtr_ori = nullptr;
	cudaMalloc(&d_imagePtr_ori, sizeof(unsigned char) * xdim * ydim * zdim);
	cudaMemcpy(d_imagePtr_ori, ImagePtr, sizeof(unsigned char) * xdim * ydim * zdim, cudaMemcpyHostToDevice);

	unsigned char* d_imagePtr_cpy = nullptr;
	cudaMalloc(&d_imagePtr_cpy, sizeof(unsigned char) * xdim * ydim * zdim);
	cudaMemcpy(d_imagePtr_cpy, ImagePtr, sizeof(unsigned char) * xdim * ydim * zdim, cudaMemcpyHostToDevice);

	int globalThreshold = 35;
	globalThreshold = 2;
	addGlobalThreshold(d_imagePtr_ori, xdim, ydim, zdim, globalThreshold);

	unsigned char* d_imagePtr = nullptr;
	cudaMalloc(&d_imagePtr, sizeof(unsigned char) * xdim * ydim * zdim);

	dim3 blockSize1(8, 8, 8);
	dim3 gridSize1((xdim + blockSize1.x - 1) / blockSize1.x,
		(ydim + blockSize1.y - 1) / blockSize1.y,
		(zdim + blockSize1.z - 1) / blockSize1.z);

	addMaxMinGlobalThreshold(d_imagePtr_cpy, xdim, ydim, zdim, globalThreshold);
	std::cerr << "set globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	/*
	//将原图中紧靠前景点的背景点赋值为1，作为之后的扩展起点
	addDarkPadding(d_imagePtr, xdim, ydim, zdim, globalThreshold);
	std::cerr << "add darkpadding cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	int* d_compress; //压缩映射矩阵
	int* d_decompress; //解压缩映射矩阵
	unsigned char* d_imagePtr_compact; //压缩后原图
	int newSize; //压缩后数组总大小

	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, xdim, ydim, zdim, newSize);

	std::cerr << "OldSize: " << xdim * ydim * zdim << " NewSize: " << newSize << std::endl;
	printf("Compress Ratio: %.2lf%%\n", 100.0 * newSize / (1.0 * xdim * ydim * zdim));

	cudaDeviceSynchronize();
	std::cerr << "compaction cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//测试1: 直接将原图变换为距离值，调用addGreyWeightTransform() 函数 将d_imagePtr_compact 变为距离变换之后的数组

	addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, xdim, ydim, zdim, newSize);
	cudaDeviceSynchronize();
	std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	//skeletonKernel << <gridSize1, blockSize1 >> > (d_imagePtr_cpy, d_imagePtr, xdim, ydim, zdim);*/

	unsigned char* d_host = (unsigned char*)malloc(sizeof(unsigned char) * xdim * ydim * zdim);
	for (int i = 0; i < 1; i++) {
		markSimpleVoxels << <gridSize1, blockSize1 >> > (d_imagePtr_cpy, xdim, ydim, zdim);
	}
	markSkeletonVoxels << <gridSize1, blockSize1 >> > (d_imagePtr_cpy, xdim, ydim, zdim);

	int blockSize2 = 256;
	int numBlocks = (xdim * ydim * zdim + 256 - 1) / 256;

	processArrayKernel << <numBlocks, blockSize2 >> > (d_imagePtr_cpy, xdim * ydim * zdim, FOREGROUND, SKEL);

	indata = ByteVolume_New(xdim, ydim, zdim);
	cudaMemcpy(indata->data, d_imagePtr_cpy, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyDeviceToHost);
	cudaFree(d_imagePtr);
	cudaFree(d_imagePtr_cpy);
}

/*************** MAIN PROGRAM *****************/


int main(int argc, const char **argv)
{
	int i;
	int infile = -1;
	int outfile = -1;
	int gammavalue = -1;
	int xdim, ydim, zdim;
	BYTE max;

	TimerClock timer;

	{


		/* Parse command line params */
		for (i = 1; i < argc; i++)
		{
			if (strcmp(argv[i], "--help") == 0) {
				printf("\nUsage: %s INFILE [-g gamma] -o OUTFILE\n", argv[0]);
				printf("INFILE is the VTK file (unsigned char ()) to use as input.\n");
				printf("gamma is a value for the pruning parameter (default=1)\n");
				printf("gamma>1: constant pruning; gamma<1: linear pruning; gamma=0: square-root pruning.\n");
				return 0;
			}
			else if (strcmp(argv[i], "-g") == 0) {
				if (i + 1 < argc) {
					gammavalue = i + 1;
					i++;
				}
				else printf("Missing value for gamma.\n");
			}
			else if (strcmp(argv[i], "-o") == 0) {
				if (i + 1 < argc) {
					outfile = i + 1;
					i++;
				}
				else printf("Missing value for output file name.\n");
			}
			else infile = i;
		}

		if (infile == -1) {
			printf("Missing input file_name. Use '%s --help' for full help.\n", argv[0]);
			return 0;
		}
		if (outfile == -1) {
			printf("Missing output file_name. Use '%s --help' for full help.\n", argv[0]);
			return 0;
		}

		if (gammavalue != -1)
			gamma_val = atof(argv[gammavalue]);

		strcpy(input_file, argv[infile]);

		strcpy(output_file, argv[outfile]);

		basefilename = basename_no_ext(input_file, "vtk");

		sprintf(skel_file, "%s_%s%d_%s%s", basefilename, "g=", gamma_val, "skel", ".vtk");
		fprintf(stdout, "gamma = %d\n", gamma_val);
		fflush(stdout);

	}

	std::cerr << "Begin " << std::endl << std::endl;
	timer.update();

	auto size = new int[3];
	unsigned char* h_inputImagePtr = loadImage(input_file, size);
	xdim = size[0];
	ydim = size[1];
	zdim = size[2];
	int allsize = xdim * ydim * zdim;

	std::cerr << "Load cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	GetSkeleton(h_inputImagePtr, xdim, ydim, zdim);

	std::cerr << "Feature Trans cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	saveTiff(output_file, indata->data, size);

	// Cleanup
	ByteVolume_Delete(indata);

	return 0;
}
