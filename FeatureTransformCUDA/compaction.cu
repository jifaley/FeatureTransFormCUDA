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

//getCompressMap:compactImage���Ӻ��������ڼ�����ѹ�����ӳ�䡣d_compressΪԪ���±�->ѹ���±꣬d_decompress��֮��
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
������compactImage
���ܣ�ѹ��ԭͼ��ȥ����0���֡� 
�����d_compactedImagePtr(ѹ�����ͼ)��d_compress (ԭͼ->ѹ��ͼӳ��)��d_decompress(ѹ��ͼ->ԭͼӳ�䣩
˼·�����Ƚ��������غ����±��Ϊtuple��������(0,value0), (1, value1), (2,value2)....
������value< 0�Ĳ���ɾ����ʣ���tuple��Ϊ: (id0, value_id0), (id1, value_id1)...
��ô,ʣ���valueֵ��Ϊѹ�����ͼ��ʣ���id��Ϊѹ�����ֵ��Ӧ��ԭͼ�е��±ꡣ
ʵ�֣�ʹ��thrust���copy_if ���� remove_if ����
*/
/*
Function��compactImage
Work��Compress the original image, leave out the zero-valued elements. (Also known as Stream Compaction)
Output��d_compactedImagePtr(The compressed image)��d_compress (The compression mapping)��d_decompress(The decompression mapping)
Implementaion��Binding the voxels and their indices to tuples, as the form of (0,value0), (1, value1), (2,value2)....
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
	int* d_sequence = d_compress; //ԭ�����������顣Ϊ�˽�ʡ��������ʱ����ͬһ��ռ�

	//������50ms���ҵ�ͬ��ʱ�䣨��ʹȥ��cuDeiveSyncronize()��cudaMemset()Ҳ��ǿ��ͬ����
	cudaDeviceSynchronize();
	std::cerr << "stream compaction preprocess cost: " << timer.getTimerMilliSec() << "ms" << std::endl;
	timer.update();


	//����copy_if��d_sequence�����µ���ԭʼ�����ݷ�0ֵ���±ꡣ�ò�����stable�ġ� newSize��Ϊ��0ֵ�ĸ�����
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

	//�����Ӧ��ӳ��
	getCompressMap << < (newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_imagePtr, d_imagePtr_compact, newSize);

	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Duing copyif " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	//�������㣬����copy_if ��getMap()��ʵ�ʺ�ʱԼ20ms����������50ms��ͬ���������ۡ�
}


__global__ void recoverKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int fullIdx = d_decompress[smallIdx];

	d_imagePtr[fullIdx] = d_imagePtr_compact[smallIdx];
}

//recoverImage:��ѹ���������ݻָ���ԭʼ����
void recoverImage(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize)
{
	assert(d_imagePtr != nullptr);
	assert(d_imagePtr_compact != nullptr);
	assert(d_decompress != nullptr);
	assert(newSize != 0);

	recoverKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_decompress, newSize);
}


