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

/*
函数：addGlobalThreshold_uppercut
功能：给d_imagePtr 指向的图像最大值截断到threshold.
*/
/*
Function：addGlobalThreshold
Work: Adding global thresholding for d_imagePtr,cut the max value of image to threshold
*/
void addGlobalThreshold_uppercut(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold)
{
	is_greater_than_th comp(threshold);
	thrust::replace_if(thrust::device, d_imagePtr, d_imagePtr + width * height * slice, d_imagePtr, comp, threshold);
}

/*
函数：addLocalThreshold_Kernel
功能：给d_imagePtr 指向的图像添加局部阈值
实现：首先根据blockSize对整个图像分块，分别统计灰度直方图。只保留块内亮度排名前5%的值。
根据：神经元信号一般相对背景来说是明亮的。
缺点：会产生较为明显的分块效应，应该添加插值等修正方法。
*/
/*
Function：addLocalThreshold_Kernel
Work：Adding local thresholding for d_imagePtr.
Implemenation: First divide the whole space into blocks according to the blockSize. The intensity histogram are calculated
in each block, and only the elements with the top 5% intensity in the block are kept.
Explaination: The intensity of neuron branch is bright in common.
Limitaion: This process may cause block effect. Users may add interpolation or other post-processing methods.
*/

__global__
void addLocalThresholdKernel(unsigned char* inputPtr, int width, int height, int slice, int blockSize, int kmax, int imax, int jmax, int* d_localThresholdArr, unsigned char globalThreshold)
{
	//valueCount: 存储灰度直方图
	__shared__ int valueCount[256];
	//valueCountCumulate: 存储直方图的前缀和
	volatile __shared__ int valueCountCumulate;
	//blockPixelCount: 小块内的像素总数
	volatile __shared__ int blockPixelCount;
	//locakThreshold: 小块内的阈值
	volatile __shared__ int localThreshold;
	volatile __shared__ int k_id, i_id, j_id, kStart, iStart, jStart;

	int bid = blockIdx.y * gridDim.x + blockIdx.x;

	if (bid >= kmax * imax * jmax) return;
	int tid = threadIdx.x;
		
	for (int i = tid; i < 256; i += blockDim.x)
	{
		valueCount[i] = 0;
	}
	__syncthreads();


	if (tid == 0)
	{
		valueCountCumulate = 0;
		k_id = bid / (imax * jmax);
		i_id = bid % (imax * jmax) / jmax;
		j_id = bid % jmax;

		kStart = k_id * blockSize;
		iStart = i_id * blockSize;
		jStart = j_id * blockSize;

		blockPixelCount = MIN(blockSize, slice - kStart) * MIN(blockSize, height - iStart) * MIN(blockSize, width - jStart);
	}

	__syncthreads();

	int temp, i, j, k;
	for (k = kStart; k < kStart + blockSize && k < slice; k++)
		for (i = iStart; i < iStart + blockSize && i < height; i++)
			for (j = jStart + tid; j < jStart + blockSize && j < width; j += blockDim.x)
			{
				temp = inputPtr[k * width * height + i * width + j];
				atomicAdd(valueCount + temp, 1);
			}

	__syncthreads();
	if (tid == 0)
	{

		valueCountCumulate = valueCount[0];
		for (int it = 1; it <= 255; it++)
		{
			valueCountCumulate += valueCount[it];
			//modified by jifaley 20240506 change 0.90 to 0.80 ->0.90 to 0.95
			if (valueCountCumulate <= blockPixelCount * 0.5)
			{
				localThreshold = it;
			}
			else
				break;
		}
		d_localThresholdArr[bid] = localThreshold;
		localThreshold = d_localThresholdArr[bid];
	}

	__syncthreads();

	for (k = kStart; k < kStart + blockSize && k < slice; k++)
		for (i = iStart; i < iStart + blockSize && i < height; i++)
			for (j = jStart + tid; j < jStart + blockSize && j < width; j += blockDim.x)
			{
				temp = inputPtr[k * width * height + i * width + j];

				//temp < 20是一个修正，不删除过亮的部分
				//modified by jifaley 20240506 change temp<=local to temp< local
				if (temp < localThreshold && temp < globalThreshold + 5)
					inputPtr[k * width * height + i * width + j] = 0;

	/*			if (temp <= localThreshold)
					inputPtr[k * width * height + i * width + j] = 0;*/
			}
}


//See addLocalThreshold_Kernel
void addLocalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, int blockSize, int globalThreahold)
{
	int kmax = (slice - 1) / blockSize + 1;
	int imax = (height - 1) / blockSize + 1;
	int jmax = (width - 1) / blockSize + 1;

	int totalBlock = kmax * imax * jmax;
	//储存每个block的局部阈值
	int* d_localThresholdArr;
	
	cudaMalloc(&d_localThresholdArr, sizeof(int) * totalBlock);
	cudaMemset(d_localThresholdArr, 0, sizeof(int) * totalBlock);

	cudaError_t errorCheck;
	//cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before Localth " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	std::cerr << "TotalBlock:"<<  totalBlock << std::endl;
	addLocalThresholdKernel << < totalBlock, 32 >> > (d_imagePtr, width, height, slice, blockSize, kmax, imax, jmax,  d_localThresholdArr, globalThreahold);


	//cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During Localth " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_localThresholdArr);
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


__global__ void medianFilterKernel(unsigned char* d_out, const unsigned char* d_in, int width, int height, int slice) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;

	if (x < width && y < height && z < slice) {
		// Create a vector to store the neighbors
		unsigned char neighbors[27];
		int count = 0;

		// Iterate over the voxel's neighborhood
		for (int dz = -1; dz <= 1; ++dz) {
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					// Compute the neighbor's index
					int nz = z + dz;
					int ny = y + dy;
					int nx = x + dx;

					// Ignore neighbors that are outside the volume
					if (nz < 0 || nz >= slice || ny < 0 || ny >= height || nx < 0 || nx >= width) {
						continue;
					}

					// Add the neighbor to the vector
					neighbors[count++] = d_in[nz * width * height + ny * width + nx];
				}
			}
		}

		// Sort the vector and set the voxel to the median value
		for (int i = 0; i < count; ++i) {
			for (int j = i + 1; j < count; ++j) {
				if (neighbors[i] > neighbors[j]) {
					unsigned char temp = neighbors[i];
					neighbors[i] = neighbors[j];
					neighbors[j] = temp;
				}
			}
		}

		d_out[z * width * height + y * width + x] = neighbors[count / 2];
	}
}

void medianFilter(unsigned char* h_in, unsigned char* h_out, int width, int height, int slice) {
	unsigned char *d_in, *d_out;

	// Allocate device memory
	cudaMalloc((void**)&d_in, width * height * slice * sizeof(unsigned char));
	cudaMalloc((void**)&d_out, width * height * slice * sizeof(unsigned char));

	// Copy data from host to device
	cudaMemcpy(d_in, h_in, width * height * slice * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// Define block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, slice);

	// Launch the kernel
	medianFilterKernel << <gridSize, blockSize >> > (d_out, d_in, width, height, slice);

	// Copy data from device to host
	cudaMemcpy(h_out, d_out, width * height * slice * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_in);
	cudaFree(d_out);
}