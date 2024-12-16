#include "compaction.h"
#include "TimerClock.hpp"

template<typename T>
struct is_non_zero {
	__host__ __device__
		bool operator()(T x) const
	{
		return x != 0;
	}
};

template<typename T>
struct is_zero {
	__host__ __device__
		bool operator()(T x) const
	{
		return x == 0;
	}
};


using thrust::placeholders::_1;

//getCompressMap:compactImage的子函数，用于计算流压缩后的映射。d_compress为元素下标->压缩下标，d_decompress反之。
//getCompressMap:The sub-function of compactImage. Calculating the mapping for stream compaction. The "d_compress" array
//is the mapping from the original element index to the compressed element index. The "d_decompress" array is the inversed mapping.
__global__
void getCompressMap(int* d_compress, int* d_decompress, unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int fullIdx = d_decompress[smallIdx];

	d_compress[fullIdx] = smallIdx;
	d_imagePtr_compact[smallIdx] = d_imagePtr[fullIdx];
}

/*
函数：compactImage
功能：压缩原图，去除非0部分。 
输出：d_compactedImagePtr(压缩后的图)，d_compress (原图->压缩图映射)，d_decompress(压缩图->原图映射）
思路：首先将所有像素和其下标绑定为tuple，类似于(0,value0), (1, value1), (2,value2)....
将所有value< 0的部分删除后，剩余的tuple即为: (id0, value_id0), (id1, value_id1)...
那么,剩余的value值即为压缩后的图，剩余的id即为压缩后的值对应的原图中的下标。
实现：使用thrust库的copy_if 或者 remove_if 操作
*/
/*
Function：compactImage
Work：Compress the original image, leave out the zero-valued elements. (Also known as Stream Compaction)
Output：d_compactedImagePtr(The compressed image)，d_compress (The compression mapping)，d_decompress(The decompression mapping)
Implementaion：Binding the voxels and their indices to tuples, as the form of (0,value0), (1, value1), (2,value2)....
After deleting the zero-valued tuples, the remainders are arranged as (id0, value_id0), (id1, value_id1)...
Thus, these values form the compressed image, and these ids are the corresponding indices in the orginal image.
This function can be implemented by thrust::copy_if or thrust::remove_if.
*/




void compactImage(unsigned char* d_imagePtr, unsigned char* &d_imagePtr_compact, int* &d_compress, int* &d_decompress, int width, int height, int slice, int& newSize)
{
	TimerClock timer;
	timer.update();

	cudaError_t errorCheck;
	cudaMalloc(&d_compress, sizeof(int) * width * height * slice);
	int* d_sequence = d_compress; //原本是两个数组。为了节省开销，暂时公用同一块空间

	//这里有50ms左右的同步时间（即使去掉cuDeiveSyncronize()，cudaMemset()也会强行同步）
	cudaDeviceSynchronize();
	std::cerr << "stream compaction preprocess cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	//经过copy_if后，d_sequence中留下的是原始体数据非0值的下标。该操作是stable的。 newSize即为非0值的个数。
	try
	{
		int* d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(width * height * slice), d_imagePtr, d_sequence, _1 != 0);
		newSize = d_copy_end - d_sequence;
	}
	catch (thrust::system_error error)
	{
		std::cerr << std::string(error.what()) << std::endl;
	}

	cudaMalloc(&d_decompress, sizeof(int) * newSize);
	cudaMalloc(&d_imagePtr_compact, sizeof(unsigned char) * newSize);
	cudaMemcpy(d_decompress, d_sequence, sizeof(int) * newSize, cudaMemcpyDeviceToDevice);
	cudaMemset(d_compress, 0xff, sizeof(int) * width * height * slice);

	//计算对应的映射
	getCompressMap << < (newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_imagePtr, d_imagePtr_compact, newSize);

	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing copyif " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	//整体运算，包括copy_if 和getMap()，实际耗时约20ms，但被上面50ms的同步严重拖累。
}


__global__ void recoverKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int fullIdx = d_decompress[smallIdx];

	d_imagePtr[fullIdx] = d_imagePtr_compact[smallIdx];
}

//recoverImage:将压缩数组内容恢复到原始数组
void recoverImage(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize)
{
	assert(d_imagePtr != nullptr);
	assert(d_imagePtr_compact != nullptr);
	assert(d_decompress != nullptr);
	assert(newSize != 0);

	recoverKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_decompress, newSize);
}


