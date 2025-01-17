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
static IntVolume*	ft,*ft_test; 
static ByteVolume*	indata, *indata_test;
static char			input_file[MAXSTR];
static char			output_file[MAXSTR];
static char*		basefilename;
static char			basename[MAXSTR];
static char			skel_file[MAXSTR];

/*************** SKELETONIZATION *****************/
__device__ void doubleScanDevice(int* ft, int lim, int ftline[],
	int dtline[], int ss[], int tt[], int interval)
{
	int q = 0, j, w;
	ss[0] = tt[0] = 0;
	for (j = 1; j < lim; j++) {
		while (q >= 0 &&
			(j - ss[q])*(j + ss[q] - 2 * tt[q]) < dtline[ss[q]] - dtline[j]) {
			q--;
		}
		if (q < 0) {
			q = 0;
			ss[0] = j;
		}
		else {
			w = 1 +
				((j + ss[q])*(j - ss[q]) + dtline[j] - dtline[ss[q]]) / (2 * (j - ss[q]));
			if (w < lim) {
				q++;
				ss[q] = j;
				tt[q] = w;
			}
		}
	}
	for (j = lim - 1; j >= 0; j--) {
		*(ft + j * interval) = ftline[ss[q]] * lim + ss[q]; /* encoding */
		if (j == tt[q]) q--;
	}
}

__global__ void featTrans1(BYTE* boundary_data, int* ftx_data, int xdim, int ydim, int zdim, int INFTY) {
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;

	int dtline[1800];
	if (y >= ydim || z >= zdim) return;

	int right = (boundary_data[xdim - 1 + y * xdim + z * xdim * ydim] == 0) ? 0 : INFTY;
	dtline[xdim - 1] = right;

	for (int x = xdim - 2; x >= 0; x--) {
		right = (boundary_data[x + y * xdim + z * xdim * ydim] == 0) ? 0 : right + 1;
		dtline[x] = right;
	}

	int left = dtline[0];
	ftx_data[y + z * ydim] = left;

	for (int x = 1; x < xdim; x++) {
		right = dtline[x];
		left = (x - left <= right) ? left : (x + right);
		ftx_data[y + z * ydim + x * zdim * ydim] = left;
	}
}

__global__ void featTrans2(int* ftxy, const int* ftx, int zdim, int xdim, int ydim) {
	int z = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.y * blockDim.y + threadIdx.y;

	if (z >= zdim || x >= xdim) return;

	int ftline[900];
	int dtline[900];
	int ss[900];
	int tt[900];

	for (int y = 0; y < ydim; y++) {
		int xy = ftx[y + z * ydim + x * zdim * ydim];
		ftline[y] = xy;
		dtline[y] = (xy - x) * (xy - x);
	}

	doubleScanDevice(&ftxy[z + x * zdim], ydim, ftline, dtline, ss, tt, zdim * xdim);
}

__global__ void featTrans3(int* ftxyz, const int* ftxy, int zdim, int xdim, int ydim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= ydim || x >= xdim) return;

	int ftline[900];
	int dtline[900];
	int ss[900];
	int tt[900];

	for (int z = 0; z < zdim; z++) {
		int xy = ftxy[z + x * zdim + y * zdim * xdim];
		ftline[z] = xy;
		dtline[z] = (xy / ydim - x) * (xy / ydim - x) + (xy % ydim - y) * (xy % ydim - y);
	}

	doubleScanDevice(&ftxyz[x + y * xdim], zdim, ftline, dtline, ss, tt, ydim * xdim);
}

IntVolume *featTrans(ByteVolume *boundary)
{
	TimerClock t;
	/* interpretation (x,y,z) in boundary IFF boundary[x][y][z] == 0
	first phase: construction of feature transform in the x-direction
	Vectors (x, y, z) are encoded as integers by
	encode(x, y, z) = z + zdim * (y + ydim * x) */
	int xdim = boundary->xdim;
	int ydim = boundary->ydim;
	int zdim = boundary->zdim;
	IntVolume *ftx, *ftxy, *ftxyz;
	int INFTY = 1 + (int)sqrt(sqr(xdim) + sqr(ydim) + sqr(zdim));
	/* The pure algorithm require a nonempty boundary; the encoding
	* requires all coordinates nonnegative. We therefore extend the
	* boundary with points with the plane x = xdim - 1 + INFTY.
	* Conditions: (xdim-1)^2 + ... + (zdim-1)^2 < INFTY^2
	* and (xdim-1+INFTY) * ydim * zdim <= Integer.MAX_VALUE */
	ftx = IntVolume_New(ydim, zdim, xdim);
	ftxyz = IntVolume_New(xdim, ydim, zdim);

	BYTE* d_boundary_data;
	int* d_ftx_data, *d_ftxyz_data, *d_ftxy_data;

	cudaMalloc(&d_boundary_data, xdim * ydim * zdim * sizeof(BYTE));
	cudaMalloc(&d_ftx_data, xdim * ydim * zdim * sizeof(int));
	cudaMalloc(&d_ftxy_data, sizeof(int) * zdim * xdim * ydim);
	cudaMalloc(&d_ftxyz_data, sizeof(int) * zdim * xdim * ydim);

	cudaMemcpy(d_boundary_data, boundary->data, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ftx_data, ftx->data, xdim * ydim * zdim * sizeof(int), cudaMemcpyHostToDevice);

	t.update();

	dim3 FTblock1(32, 32);
	dim3 FTgrid1((ydim + FTblock1.x - 1) / FTblock1.x, (zdim + FTblock1.y - 1) / FTblock1.y);

	featTrans1 << <FTgrid1, FTblock1 >> > (d_boundary_data, d_ftx_data, xdim, ydim, zdim, INFTY);

	cudaFree(d_boundary_data);

	cudaDeviceSynchronize();
	
	std::cerr << "featTrans 1 cost " << t.getTimerMilliSec() << "ms" << std::endl << std::endl;
	t.update();
	

	/* second phase: construction of feature transform in the xy-direction
	* based on feature transform in the x-direction */
	
	dim3 FTblock2(32, 32);
	dim3 FTgrid2((zdim + FTblock2.x - 1) / FTblock2.x, (xdim + FTblock2.y - 1) / FTblock2.y);

	featTrans2 << <FTgrid2, FTblock2 >> > (d_ftxy_data, d_ftx_data, zdim, xdim, ydim);

	cudaDeviceSynchronize();
	

	std::cerr << "featTrans 2 cost " << t.getTimerMilliSec() << "ms" << std::endl << std::endl;
	t.update();

	/* third phase: construction of feature transform in the xyz-direction
	* based on feature transform in the xy-direction */
	

	dim3 FTblock3(32, 32);
	dim3 FTgrid3((xdim + FTblock3.x - 1) / FTblock3.x, (ydim + FTblock3.y - 1) / FTblock3.y);

	featTrans3 << <FTgrid3, FTblock3 >> > (d_ftxyz_data, d_ftxy_data, zdim, xdim, ydim);

	cudaDeviceSynchronize();
	
	std::cerr << "featTrans 3 cost " << t.getTimerMilliSec() << "ms" << std::endl << std::endl;
	t.update();

	cudaMemcpy(ftxyz->data, d_ftxyz_data, sizeof(int) * zdim * xdim * ydim, cudaMemcpyDeviceToHost);

	std::cerr << "featTrans 3copy cost " << t.getTimerMilliSec() << "ms" << std::endl << std::endl;
	t.update();

	IntVolume_Delete(ftx);
	cudaFree(d_ftx_data);
	cudaFree(d_ftxy_data);
	cudaFree(d_ftxyz_data);

	return ftxyz;
}

IntVolume* NewfeatTrans(ByteVolume* indata) 
{
	TimerClock timer;
	int width = indata->xdim;
	int height = indata->ydim;
	int slice = indata->zdim;
	unsigned char* d_imagePtr = nullptr;
	cudaMalloc(&d_imagePtr, sizeof(unsigned char) * width * height * slice);
	cudaMemcpy(d_imagePtr, indata->data, sizeof(unsigned char) * width * height * slice, cudaMemcpyHostToDevice);
	
	int globalThreshold = 2;
	addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);
	std::cerr << "set globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	cudaMemcpy(indata->data, d_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyDeviceToHost);

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


	//测试1: 直接将原图变换为距离值，调用addGreyWeightTransform() 函数 将d_imagePtr_compact 变为距离变换之后的数组

	addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
	cudaDeviceSynchronize();
	std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//调用recoverImage() 将d_imagePtr_compact中的数据解压缩到原图d_imagePtr中
	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	//int* d_parentPtr_compact;
	//unsigned char* d_activeMat_compact;
	//cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);
	//cudaMalloc(&d_activeMat_compact, sizeof(unsigned char) * newSize);
	//cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize);
	//cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);


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

	//int* d_ftarr;
	//cudaMalloc(&d_ftarr, sizeof(int) * width * height * slice);

	////ftarr数组即为IMA3D 要求的，距离最近背景点的位置。根据之前计算的parent信息倒着跟踪计算得到
	//findFtPoints(d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);

	//cudaDeviceSynchronize();
	//std::cerr << "findFtPoints cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();

	//int* h_ftarr = (int*)malloc(sizeof(int) * width * height * slice);
	//cudaMemcpy(h_ftarr, d_ftarr, sizeof(int) * width * height * slice, cudaMemcpyDeviceToHost);

	IntVolume* result = IntVolume_New(width, height, slice);
	//result->data = h_ftarr;
	//result->data = d_imagePtr;
	cudaFree(d_imagePtr);
	cudaFree(d_compress);
	cudaFree(d_decompress);
	cudaFree(d_imagePtr_compact);
	//cudaFree(d_parentPtr_compact);
	//cudaFree(d_activeMat_compact);
	//cudaFree(d_ftarr);
	return result;
}

IntVolume* CDTfeatTrans(ByteVolume* indata)
{
	int xdim = indata->xdim;
	int ydim = indata->ydim;
	int zdim = indata->zdim;

	IntVolume* CDT = IntVolume_New(xdim, ydim, zdim);

	for (int z = 0; z < zdim; z++) {
		for (int y = 0; y < ydim; y++) {
			for (int x = 0; x < xdim; x++) {
				int cdt = 0;
				int cdt_x = 0;
				int cdt_y = 0;
				int cdt_z = 0;
				for (int i = 0; i < min(x, xdim - x); i++) {
					if ((int)indata->data[x + i + y * xdim + z * xdim * ydim] == 0 ||
						(int)indata->data[x - i + y * xdim + z * xdim * ydim] == 0) {
						cdt_x = i;
						break;
					}	
				}
				for (int i = 0; i < min(y, ydim - y); i++) {
					if ((int)indata->data[x + (y + i) * xdim + z * xdim * ydim] == 0 ||
						(int)indata->data[x + (y - i) * xdim + z * xdim * ydim] == 0) {
						cdt_y = i;
						break;
					}
				}
				for (int i = 0; i < min(z, zdim - z); i++) {
					if ((int)indata->data[x + y * xdim + (z + i) * xdim * ydim] == 0 ||
						(int)indata->data[x + y * xdim + (z - i) * xdim * ydim] == 0) {
						cdt_z =i;
						break;
					}
				}
				cdt = min(cdt_x, min(cdt_y, cdt_z));
				CDT->data[x + y * xdim + z * xdim * ydim] = cdt;
			}
		}
	}
	return CDT;
}

__device__ void compare(BYTE* xskel, BYTE* pskel,
	int x, int y, int z,
	int p, int q, int r,
	int xf, int pf, int yf, int qf, int zf, int rf,
	int ydim, int zdim, float gamma_val) {
	int dif = (xf - pf) * (xf - pf) + (yf - qf) * (yf - qf) + (zf - rf) * (zf - rf);

	if (dif > 1 && dif > (gamma_val > 0 ? gamma_val * gamma_val
		: gamma_val < 0 ? ((x - xf + p - pf) * (x - xf + p - pf) +
			(y - yf + q - qf) * (y - yf + q - qf) +
			(z - zf + r - rf) * (z - zf + r - rf)) * gamma_val * gamma_val
		: sqrtf((x - xf + p - pf) * (x - xf + p - pf) +
			(y - yf + q - qf) * (y - yf + q - qf) +
			(z - zf + r - rf) * (z - zf + r - rf)) +
		2 * ((x - p) * (xf - pf) + (y - q) * (yf - qf) + (z - r) * (zf - rf)) + 1.5)) {
		int crit = (xf - pf) * (xf + pf - x - p) + (yf - qf) * (yf + qf - y - q) + (zf - rf) * (zf + rf - z - r);
		if (crit >= 0) *xskel = SKEL;
		if (crit <= 0) *pskel = SKEL;
	}
}


__global__ void skeletonKernel(BYTE* skel, int* ft, int xdim, int ydim, int zdim, float gamma_val) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < xdim && y < ydim && z < zdim) {
		int idx = x + y * xdim + z * xdim * ydim;

		if (x >= 1) {
			int leftIdx = (x - 1) + y * xdim + z * xdim * ydim;
			if (skel[idx] == FOREGROUND || skel[leftIdx] == FOREGROUND) {
				int xf = ft[idx];
				int pf = ft[leftIdx];
				int zf = xf % zdim; xf /= zdim;
				int yf = xf % ydim; xf /= ydim;
				int rf = pf % zdim; pf /= zdim;
				int qf = pf % ydim; pf /= ydim;
				compare(&skel[idx], &skel[leftIdx], x, y, z, x - 1, y, z, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
		if (y >= 1) {
			int topIdx = x + (y - 1) * xdim + z * xdim * ydim;
			if (skel[idx] == FOREGROUND || skel[topIdx] == FOREGROUND) {
				int xf = ft[idx];
				int pf = ft[topIdx];
				int zf = xf % zdim; xf /= zdim;
				int yf = xf % ydim; xf /= ydim;
				int rf = pf % zdim; pf /= zdim;
				int qf = pf % ydim; pf /= ydim;
				compare(&skel[idx], &skel[topIdx], x, y, z, x, y - 1, z, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
		if (z >= 1) {
			int frontIdx = x + y * xdim + (z - 1) * xdim * ydim;
			if (skel[idx] == FOREGROUND || skel[frontIdx] == FOREGROUND) {
				int xf = ft[idx];
				int pf = ft[frontIdx];
				int zf = xf % zdim; xf /= zdim;
				int yf = xf % ydim; xf /= ydim;
				int rf = pf % zdim; pf /= zdim;
				int qf = pf % ydim; pf /= ydim;
				compare(&skel[idx], &skel[frontIdx], x, y, z, x, y, z - 1, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
	}
}

__global__ void NewskeletonKernel(BYTE* skel, int* ft, int xdim, int ydim, int zdim, float gamma_val) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < xdim && y < ydim && z < zdim) {
		int idx = x + y * xdim + z * xdim * ydim;

		if (x >= 1) {
			int leftIdx = (x - 1) + y * xdim + z * xdim * ydim;
			if (skel[idx] >= FOREGROUND || skel[leftIdx] >= FOREGROUND) {
				int zf = ft[idx];
				int rf = ft[leftIdx];
				int xf = zf % xdim; zf /= xdim;
				int yf = zf % ydim; zf /= ydim;
				int pf = rf % xdim; rf /= xdim;
				int qf = rf % ydim; rf /= ydim;
				compare(&skel[idx], &skel[leftIdx], x, y, z, x - 1, y, z, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
		if (y >= 1) {
			int topIdx = x + (y - 1) * xdim + z * xdim * ydim;
			if (skel[idx] >= FOREGROUND || skel[topIdx] >= FOREGROUND) {
				int zf = ft[idx];
				int rf = ft[topIdx];
				int xf = zf % xdim; zf /= xdim;
				int yf = zf % ydim; zf /= ydim;
				int pf = rf % xdim; rf /= xdim;
				int qf = rf % ydim; rf /= ydim;
				compare(&skel[idx], &skel[topIdx], x, y, z, x, y - 1, z, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
		if (z >= 1) {
			int frontIdx = x + y * xdim + (z - 1) * xdim * ydim;
			if (skel[idx] >= FOREGROUND || skel[frontIdx] >= FOREGROUND) {
				int zf = ft[idx];
				int rf = ft[frontIdx];
				int xf = zf % xdim; zf /= xdim;
				int yf = zf % ydim; zf /= ydim;
				int pf = rf % xdim; rf /= xdim;
				int qf = rf % ydim; rf /= ydim;
				compare(&skel[idx], &skel[frontIdx], x, y, z, x, y, z - 1, xf, pf, yf, qf, zf, rf, ydim, zdim, gamma_val);
			}
		}
	}
}

void CDT2MAT(BYTE* skel, int* dis, int xdim, int ydim, int zdim)
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

void runSkeletonization(IntVolume* ft, ByteVolume* skel, float gamma_val) {
	int xdim = ft->xdim;
	int ydim = ft->ydim;
	int zdim = ft->zdim;

	BYTE* d_skel;
	int* d_ft;
	cudaMalloc(&d_skel, xdim * ydim * zdim * sizeof(BYTE));
	cudaMalloc(&d_ft, xdim * ydim * zdim * sizeof(int));

	cudaMemcpy(d_skel, skel->data, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ft, ft->data, xdim * ydim * zdim * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockSize(8, 8, 8);
	dim3 gridSize((xdim + blockSize.x - 1) / blockSize.x,
		(ydim + blockSize.y - 1) / blockSize.y,
		(zdim + blockSize.z - 1) / blockSize.z);

	skeletonKernel << <gridSize, blockSize >> > (d_skel, d_ft, xdim, ydim, zdim, gamma_val);

	cudaMemcpy(skel->data, d_skel, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyDeviceToHost);

	cudaFree(d_skel);
	cudaFree(d_ft);
}

void NewrunSkeletonization(IntVolume* ft, ByteVolume* skel, float gamma_val) {
	int xdim = ft->xdim;
	int ydim = ft->ydim;
	int zdim = ft->zdim;

	BYTE* d_skel;
	int* d_ft;
	cudaMalloc(&d_skel, xdim * ydim * zdim * sizeof(BYTE));
	cudaMalloc(&d_ft, xdim * ydim * zdim * sizeof(int));

	cudaMemcpy(d_skel, skel->data, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ft, ft->data, xdim * ydim * zdim * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockSize(8, 8, 8);
	dim3 gridSize((xdim + blockSize.x - 1) / blockSize.x,
		(ydim + blockSize.y - 1) / blockSize.y,
		(zdim + blockSize.z - 1) / blockSize.z);

	NewskeletonKernel << <gridSize, blockSize >> > (d_skel, d_ft, xdim, ydim, zdim, gamma_val);

	cudaMemcpy(skel->data, d_skel, xdim * ydim * zdim * sizeof(BYTE), cudaMemcpyDeviceToHost);

	cudaFree(d_skel);
	cudaFree(d_ft);
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

template <typename T>
int compareArrays(const T* arr1, const T* arr2, size_t size) {
	int count = 0;

	for (size_t i = 0; i < size; ++i) {
		if (arr1[i] != arr2[i]) {
			++count;
		}
	}

	return count;
}

int compareFTs(int* arr1, int* arr2, int xdim, int ydim, int zdim, unsigned char* origin) {
	int count = 0;
	int number = 0;
	for (int i = 0; i < xdim * ydim * zdim; ++i) {
		if ((int)origin[i] == 1) {
			// 新版本FT: arr1[i] = z * xdim * ydim + y * xdim + x
			// 旧版本FT: arr2[i] = (y + x * ydim) * zdim + z = x * ydim * zdim + y * zdim + z
			int idx1 = arr1[i];
			int x1 = idx1 % xdim; idx1 /= xdim;
			int y1 = idx1 % ydim; idx1 /= ydim;
			int z1 = idx1;

			int idx2 = arr2[i];
			int z2 = idx2 % zdim; idx2 /= zdim;
			int y2 = idx2 % ydim; idx2 /= ydim;
			int x2 = idx2;

			int idx = i;
			int x3 = idx % xdim; idx /= xdim;
			int y3 = idx % ydim; idx /= ydim;
			int z3 = idx;

			int dist1 = (x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3) + (z1 - z3) * (z1 - z3);
			int dist2 = (x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3) + (z2 - z3) * (z2 - z3);

			if (dist1 < dist2) {
				count++;
				if (number < 10) {
					cout << x1 << " " << y1 << " " << z1 << endl;
					cout << x2 << " " << y2 << " " << z2 << endl;
					cout << x3 << " " << y3 << " " << z3 << endl;
					cout << endl;
					number++;
				}
			}
		}	
	}

	return count;
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

	unsigned char* h_inputImagePtr_test = loadImage(input_file, size);

	std::cerr << "Load cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	dealWithInput(h_inputImagePtr, allsize, max);

	indata = ByteVolume_New(xdim, ydim, zdim);
	indata->data = h_inputImagePtr;

	indata_test = ByteVolume_New(xdim, ydim, zdim);
	indata_test->data = h_inputImagePtr_test;

	
	//ft = featTrans(indata);
	ft_test = NewfeatTrans(indata_test);
	ft = CDTfeatTrans(indata);

	/*int count1 = compareArrays(indata->data, indata_test->data, allsize);
	cout << "count1: " << count1 << endl;*/

	/*int count2 = compareFTs(ft_test->data, ft->data, xdim, ydim, zdim, h_inputImagePtr);
	cout << "count2: " << count2 << endl;*/

	std::cerr << "Feature Trans cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//memcpy(indata_test->data, indata->data, sizeof(unsigned char)* xdim* ydim* zdim);

	// Construct the skeleton from the feature transform into 'indata'
	//runSkeletonization(ft, indata, gamma_val);
	//NewrunSkeletonization(ft_test, indata_test, gamma_val);
	//CDT2MAT(indata->data, indata_test->data, xdim, ydim, zdim);
	CDT2MAT(indata->data, ft->data, xdim, ydim, zdim);

	//int count3 = compareArrays(indata->data, indata_test->data, allsize);
	//cout << "count3: " << count3 << endl;

	std::cerr << "Skeleton cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	finalProcess(indata, allsize);
	//finalProcess(indata_test, allsize);

	//int count4 = compareArrays(indata->data, indata_test->data, allsize);
	//cout << "count4: " << count4 << endl;

	std::cerr << "final cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	saveTiff(output_file, indata->data, size);

	// Cleanup
	//ByteVolume_Delete(indata);
	//IntVolume_Delete(ft);

	return 0;
}
