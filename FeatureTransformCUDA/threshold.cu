#include "threshold.h"

static __constant__ const int dx3dconst[6] = { -1, 1, 0, 0, 0, 0 };
static __constant__ const int dy3dconst[6] = { 0, 0, -1, 1, 0, 0 };
static __constant__ const int dz3dconst[6] = { 0, 0, 0, 0, -1, 1 };

static __constant__ const int dx3d26const[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
static __constant__ const int dy3d26const[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
static __constant__ const int dz3d26const[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };


struct is_less_than_th
{
	is_less_than_th(unsigned char th = 0) :_th(th){}
	__host__ __device__
		bool operator()(int x)
	{
		return x < _th;
	}
private:
	unsigned char _th;
};

struct is_greater_than_th
{
	is_greater_than_th(unsigned char th = 0) :_th(th) {}
	__host__ __device__
		bool operator()(int x)
	{
		return x > _th;
	}
private:
	unsigned char _th;
};

struct getVar : thrust::unary_function<unsigned char, double>
{
	getVar(double mean) : _mean(mean){}
	const double _mean;
	__host__ __device__ double operator()(unsigned char data) const
	{
		return (data - _mean) * (data - _mean);
	}
};

using thrust::placeholders::_1;

int getGlobalThreshold(unsigned char* h_imagePtr, unsigned char* d_imagePtr, int width, int height, int slice)
{
	double sum = 0;
	//for (int i = 0; i < width * height * slice; i++)
	//{
	//	sum += h_imagePtr[i];
	//}

	sum = thrust::reduce(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, 0.0, thrust::plus<double>());

	double mean = sum / (width * height * slice);

	double var = 0;


	var = thrust::transform_reduce(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, getVar(mean), 0.0, thrust::plus<double>());
	

	var = var / (width * height * slice - 1);


	double std = sqrt(var);

	double td = (std < 10) ? 10 : std;

	printf("mean: %.2lf, std:%.2lf\n", mean, std);

	int th = mean + 0.5 * td;

	printf("autoset global Th = %d\n", th);

	return th;

}


/*
函数：addGlobalThreshold
功能：给d_imagePtr 指向的图像添加全局阈值
*/
/*
Function：addGlobalThreshold
Work: Adding global thresholding for d_imagePtr.
*/
void addGlobalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold)
{
	is_less_than_th comp(threshold);
	thrust::replace_if(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, d_imagePtr, comp, 0);
}

void addMaxMinGlobalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold)
{
	is_less_than_th comp(threshold);
	thrust::replace_if(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, d_imagePtr, comp, 0);

	is_greater_than_th comp2(1);
	thrust::replace_if(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, d_imagePtr, comp2, 1);
}



/*
函数：addDarkPaddingKernel
功能：给d_imagePtr 指向的图像进行补充
实现：对于足够亮的区域，将其周边的暗区灰度置为1
根据：试图填补不同亮区之间的缝隙，使得后面追踪时能成功连接相邻的亮区
*/
/*
Function：addDarkPaddingKernel
Work：padding the image.
Implemenation：For each bright area, set its neighboring background pixel's intensity from 0 to 1.
Explaination: Try to fill the holes and gaps between different bright areas. The 0-valued elements will be removed
in the later processes, but the 1-valued will not.
*/
__global__
void addDarkPaddingKernel(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= width * height * slice) return;

	unsigned char curValue = d_imagePtr[idx];
	if (curValue >= threshold)
	{
		int3 curPos;
		curPos.z = idx / (width * height);
		curPos.y = idx % (width * height) / width;
		curPos.x = idx % width;
		//printf("%d %d %d\n", curPos.x, curPos.y, curPos.z);

		int3 neighborPos;
		int neighborIdx;
		for (int k = 0; k < 26; k++)
		{
			neighborPos.x = curPos.x + dx3d26const[k];
			neighborPos.y = curPos.y + dy3d26const[k];
			neighborPos.z = curPos.z + dz3d26const[k];
			if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
				|| neighborPos.z < 0 || neighborPos.z >= slice)
				continue;
			neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
			if (d_imagePtr[neighborIdx] == 0)
			{
				d_imagePtr[neighborIdx] = 1;
			}
		}

		int windowSize = 1;
		//windowSize = 5;
		//windowSize = 3; //for flycircuits

		for (int dx = -windowSize; dx <= windowSize; dx++)
			for (int dy = -windowSize; dy <= windowSize; dy++)
				for (int dz = -windowSize; dz <= windowSize; dz++)
				{
					neighborPos.x = curPos.x + dx;
					neighborPos.y = curPos.y + dy;
					neighborPos.z = curPos.z + dz;
					if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
						|| neighborPos.z < 0 || neighborPos.z >= slice)
						continue;
					neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
					if (d_imagePtr[neighborIdx] == 0)
					{
						d_imagePtr[neighborIdx] = 1;
					}
				}
	}
}


//See addDarkPaddingKernel
void addDarkPadding(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold)
{
	addDarkPaddingKernel << <(width * height * slice - 1) / 256 + 1, 256 >> > (d_imagePtr, width, height, slice, threshold);
}

