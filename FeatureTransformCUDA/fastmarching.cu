#include "fastmarching.h"
#include "TimerClock.hpp"

using namespace cooperative_groups;



//函数：gwdtExtendKernel
//功能:通过最短路的方法计算GreyWeight Distance Transform (参见APP2)。本Kernel用于将某个像素向邻居扩展。
//Work:Calculating the GreyWeight Distance Transform (See Xiao et al. APP2: automatic tracing of 3d neuron morphology...)
//This kernel is used for extending the fast-marching frontier to neighbors.
__global__
void gwdtExtendKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, int* d_distPtr, int* d_updateDistPtr, unsigned char* d_inCurFrontier, unsigned char* d_inNextFrontier,  int width, int height, int slice, int newSize, int compact_size)
{
	//smallIdx: 压缩后的下标 fullIdx: 原始图像的下标 newSize: 压缩后图像的大小
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= compact_size) return;

	int smallIdx = d_frontier_compact[tid];
	{
		d_inCurFrontier[smallIdx] = 0;
		if (smallIdx >= newSize) return;

		//判断该点是否刚被更新过

		int fullIdx = d_decompress[smallIdx];

		int3 curPos;
		curPos.z = fullIdx / (width * height);
		curPos.y = fullIdx % (width * height) / width;
		curPos.x = fullIdx % width;

		int3 neighborPos;
		int neighborIdx, neighborSmallIdx;
		int neighborValue;

		int curDist = d_distPtr[smallIdx];

		for (int k = 0; k < 6; k++)
		{
			neighborPos.x = curPos.x + dx3dconst[k];
			neighborPos.y = curPos.y + dy3dconst[k];
			neighborPos.z = curPos.z + dz3dconst[k];
			if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
				|| neighborPos.z < 0 || neighborPos.z >= slice)
				continue;
			neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;
			neighborValue = d_imagePtr[neighborIdx];

			//若邻居为背景则不扩展
			//When neighboring value is background
			if (neighborValue == 0) continue;

			//最短路计算方法：从所有背景像素出发，边的长度为像素值
			//The edge length for the shortest path algorithm is the intensity of the voxels.
			neighborSmallIdx = d_compress[neighborIdx];
			int old = atomicMin(&d_updateDistPtr[neighborSmallIdx], curDist + neighborValue);

			//old返回的是原子操作之前的值。如果发现updateDist数组被更新了，激活该点。
			//The "old" variable denotes the value before the atomic operation. If the "updateDist" is updated, activate this point. 
			if (curDist + neighborValue < old)
				d_inNextFrontier[neighborSmallIdx] = 1;
				//d_nextStatus[neighborSmallIdx] = TRIAL;
		}
	}
}


//函数：gwdtUpdateKernel
//功能:通过最短路的方法计算GreyWeight Distance Transform。本Kernel用于更新节点的dist值。
//Work:Calculating the GreyWeight Distance Transform (See Xiao et al. APP2: automatic tracing of 3d neuron morphology...)
//This kernel is used for updating the distance value of nodes.
__global__
void gwdtUpdateKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_frontier_compact, int* d_distPtr, int* d_updateDistPtr, unsigned char* d_inCurFrontier, unsigned char* d_inNextFrontier,  int width, int height, int slice, int newSize, int compact_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= compact_size) return;
	int smallIdx = d_frontier_compact[tid];

	if (!d_inNextFrontier[smallIdx]) return;
	d_inNextFrontier[smallIdx] = 0;
	//更新阶段到下一个锋面。本像素原来是上一锋面的ExtendKernel()扩展出来的。
	//Walking into the next frontier. The current pixel is generated by the gwdtExtendKernel() of the previous frontier.
	d_inCurFrontier[smallIdx] = 1;

	int updateValue = d_updateDistPtr[smallIdx];
	int curValue = d_distPtr[smallIdx];

	if (updateValue < curValue)
	{
		d_distPtr[smallIdx] = updateValue;
	}
}


//预处理：把所有临近0的像素作为起始点，他们距离背景的距离即为他们的像素值
//Preprocessing. Putting all of the voxels who have a background voxel as neighbor into the starting frontier.
//The intital distance value is set as their voxel intensity.
__global__ void gwdtPreProcessKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_compress, int* d_decompress,  int* d_distPtr, int* d_updateDistPtr,  unsigned char* d_inCurFrontier, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int3 neighborPos;
	int3 curPos;
	unsigned char neighborValue;
	unsigned char curValue;

	curValue = d_imagePtr_compact[smallIdx];
	int fullIdx = d_decompress[smallIdx];
	int neighborIdx;

	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	
	for (int k = 0; k < 6; k++)
	{
		neighborPos.x = curPos.x + dx3dconst[k];
		neighborPos.y = curPos.y + dy3dconst[k];
		neighborPos.z = curPos.z + dz3dconst[k];
		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			continue;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];

		if (neighborValue == 0)
		{
			d_distPtr[smallIdx] = curValue;
			d_updateDistPtr[smallIdx] = curValue;
			d_inCurFrontier[smallIdx] = 1;
			break;
		}
	}
}


//后处理：将距离变换的结果缩放到0-255
//Postprocessing: Mapping the result of the GreyWeight Distance Transform to [0,255].
__global__ void gwdtPostProcessKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_distPtr, float maxValue, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	float temp = d_distPtr[smallIdx];
	int fullIdx = d_decompress[smallIdx];
	d_imagePtr_compact[smallIdx] = temp / maxValue * 255;
	d_imagePtr[fullIdx] = temp / maxValue * 255;
}


/*
函数：addGreyWeightTransform
功能：通过最短路迭代，找到每个像素到离他最近的背景像素的距离（点之间距离即为像素值），
然后根据这个距离对原图进行映射。这样映射之后，距离背景更远的像素会更亮（也更有可能是神经纤维的中心），
处理过后，下一步的追踪更容易沿着神经纤维中心拓展。
输出：d_imagePtr （直接在原图上进行改动）
注意：这里的addGreyWeightTransform() 和 下面的BuildInitNeuron()都使用了并行fast-marching作为核心。
*/
/*
Function：addGreyWeightTransform
Work：The GPU implementation of the Grey Weighted Distance Transform. (For GWDT, See Xiao et al. APP2: automatic tracing of 3d neuron morphology...)
After this transfomation, the intensity of voxels next to the background will decrease, and the intensity of voxels located in center of neurons will increase.  
Output：d_imagePtr (Directly making modifications on the original image)
Note: Both the addGreyWeightTransform() function and the following BuildInitNeuron() function are based on parallel fast-marching.
*/

void addGreyWeightTransform(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_compress, int* d_decompress, int width, int height, int slice, int newSize)
{
	int* d_distPtr;
	int* d_updateDistPtr;


	cudaMalloc((void**)&d_distPtr, sizeof(int) * newSize);
	cudaMalloc((void**)&d_updateDistPtr, sizeof(int) * newSize);

	thrust::device_ptr<int> d_distPtr_thrust(d_distPtr);
	thrust::device_ptr<int> d_updateDistPtr_thrust(d_updateDistPtr);
	thrust::fill(d_distPtr_thrust, d_distPtr_thrust + newSize, 100000000);
	thrust::fill(d_updateDistPtr_thrust, d_updateDistPtr_thrust + newSize, 100000000);

	//thrust::fill(d_curStatus_thrust, d_curStatus_thrust + width * height * slice, FARAWAY);

	unsigned char* d_inCurFrontier;
	unsigned char* d_inNextFrontier;
	int* d_frontier_compact;

	cudaMalloc((void**)&d_inNextFrontier, sizeof(unsigned char) * newSize);
	cudaMalloc((void**)&d_inCurFrontier, sizeof(unsigned char) * newSize);
	cudaMalloc((void**)&d_frontier_compact, sizeof(int) * newSize);

	cudaMemset(d_inNextFrontier, 0, sizeof(unsigned char) * newSize);
	cudaMemset(d_inCurFrontier, 0, sizeof(unsigned char) * newSize);
	cudaMemset(d_frontier_compact, 0, sizeof(int) * newSize);

	//thrust::device_vector<int>dv_frontier_compact(newSize);
	//int* d_frontier_compact = thrust::raw_pointer_cast(dv_frontier_compact.data());


	cudaError_t errorCheck;

	gwdtPreProcessKernel << < (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_distPtr, d_updateDistPtr, d_inCurFrontier, width, height, slice, newSize);
	
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In GWDT Preprocess: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	using thrust::placeholders::_1;

	int* d_copy_end;
	int compact_size;


	//流压缩stream compaction,将等于0的部分删除
	//Stream Compaction, removing the zero-valued elements
	try
	{
		d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_inCurFrontier, d_frontier_compact, _1 != 0);
		compact_size = d_copy_end - d_frontier_compact;
	}
	catch (thrust::system_error error)
	{
		std::cerr << std::string(error.what()) << std::endl;
	}

	printf("GWDT start frontier size: %d\n", compact_size);


	//blockSize:64
	//maxBlockNum:512
	int counter = 0;
	while (1)
	{
		counter++;
		gwdtExtendKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_frontier_compact,  d_compress, d_decompress,  d_distPtr, d_updateDistPtr, d_inCurFrontier, d_inNextFrontier,

			width, height, slice, newSize, compact_size);


		d_copy_end = thrust::copy_if(thrust::device, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(newSize), d_inNextFrontier, d_frontier_compact, _1 != 0);
		compact_size = d_copy_end - d_frontier_compact;
		if (compact_size == 0)
			break;

		gwdtUpdateKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_frontier_compact, d_distPtr, d_updateDistPtr, d_inCurFrontier, d_inNextFrontier,

			width, height, slice, newSize, compact_size);
	}


	int maxValue = thrust::reduce(d_distPtr_thrust, d_distPtr_thrust + newSize, 0, thrust::maximum<int>());
	std::cerr << "Max value by reduce: " << maxValue << std::endl;

	gwdtPostProcessKernel << <(newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_distPtr, maxValue, width, height, slice, newSize);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In GWDT PostProcess: " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_distPtr);
	cudaFree(d_updateDistPtr);
	cudaFree(d_inCurFrontier);
	cudaFree(d_inNextFrontier);
	cudaFree(d_frontier_compact);
}


//流压缩后的数组长度
__device__ int d_compact_size;


//预处理：把所有临近0的像素作为起始点，他们距离背景的距离即为他们的像素值
//Preprocessing. Putting all of the voxels who have a background voxel as neighbor into the starting frontier.
//The intital distance value is set as their voxel intensity.
__global__ void ftPreProcessKernel(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_compress, int* d_decompress, float* d_distPtr, float* d_updateDistPtr, int* d_frontier_compact, unsigned char* d_activeMat_compact, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int3 neighborPos;
	int3 curPos;
	unsigned char neighborValue;
	unsigned char curValue;

	curValue = d_imagePtr_compact[smallIdx];
	int fullIdx = d_decompress[smallIdx];
	int neighborIdx;

	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;


	for (int k = 0; k < 6; k++)
	{
		neighborPos.x = curPos.x + dx3dconst[k];
		neighborPos.y = curPos.y + dy3dconst[k];
		neighborPos.z = curPos.z + dz3dconst[k];
		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			continue;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];

		if (neighborValue == 0)
		{
			d_distPtr[smallIdx] = 1;
			d_updateDistPtr[smallIdx] = 1;
			//上面说的是distance = 1，下面说的是在frontier中


			d_activeMat_compact[smallIdx] = ALIVE;
			d_frontier_compact[smallIdx] = smallIdx;

			break;
		}
	}
}


__global__
void ftTracingExtendKernel_warpShuffle_atomic(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_frontier_compact, int* d_compress, int* d_decompress, float* d_distPtr, float* d_updateDistPtr,
	int width, int height, int slice, int newSize, int compact_size, int* d_frontier_compact_2, unsigned char* d_activeMat_compact)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int blockLength;
	__shared__ int blockOffset;

	int warp_id = threadIdx.x / 32;
	int lane_id = threadIdx.x % 32;
	int pointId = blockIdx.x * blockDim.x / 32 + warp_id;

	if (pointId >= compact_size) return;

	int smallIdx, fullIdx;
	smallIdx = d_frontier_compact[pointId];
	if (d_activeMat_compact[smallIdx] != ALIVE) return;


	int curValue;
	float curDist;

	if (lane_id == 0)
	{
		smallIdx = d_frontier_compact[pointId];
		curValue = d_imagePtr_compact[smallIdx];
		curDist = d_distPtr[smallIdx];
		fullIdx = d_decompress[smallIdx];
	}

	//Set initial value. The __shfl_sync function transmit(broadcast) values in the warp.
	fullIdx = __shfl_sync(-1, fullIdx, 0);
	curDist = __shfl_sync(-1, curDist, 0);
	curValue = __shfl_sync(-1, curValue, 0);

	int3 curPos;
	curPos.z = fullIdx / (width * height);
	curPos.y = fullIdx % (width * height) / width;
	curPos.x = fullIdx % width;

	int3 neighborPos;
	int neighborIdx;
	int neighborSmallIdx;
	unsigned char neighborValue;

	int k = lane_id;
	int modified = 0;

	if (k < 26)
	{
		neighborPos.x = curPos.x + dx3d26const[k];
		neighborPos.y = curPos.y + dy3d26const[k];
		neighborPos.z = curPos.z + dz3d26const[k];

		if (neighborPos.x < 0 || neighborPos.x >= width || neighborPos.y < 0 || neighborPos.y >= height
			|| neighborPos.z < 0 || neighborPos.z >= slice)
			return;
		neighborIdx = neighborPos.z * width * height + neighborPos.y * width + neighborPos.x;

		neighborValue = d_imagePtr[neighborIdx];


		neighborSmallIdx = d_compress[neighborIdx];
		if (neighborSmallIdx == -1) return;

		neighborValue = d_imagePtr_compact[neighborSmallIdx];
		if (neighborValue == 0) return;

		float EuclidDist = 1;
		//EuclidDist = sqrtf(dx3d26const[k] * dx3d26const[k] + dy3d26const[k] * dy3d26const[k] + dz3d26const[k] * dz3d26const[k]);
		//由于只有26个邻居，直接把对应的欧式距离存储起来了
		////The distance of neighbors are stored at the constant memory.
		EuclidDist = EuclidDistconst[k];
		//两点之间的dist根据两点的欧式距离和亮度计算
		//float deltaDist = gwdtFunc_gpu(EuclidDist, curValue, neighborValue);

		if (neighborSmallIdx < 0)
			printf("neighbor < 0!\n");

		float deltaDist;
		deltaDist = EuclidDist;




		//在使用原子操作之前，进行一次快速检查。如果当前点连上阶段的邻居都更新不了，就放弃更新
		//A fast check before atomic operations.
		if (d_distPtr[neighborSmallIdx] - 1e-5 < curDist + deltaDist)
			return;

		float newDist = curDist + deltaDist;
		//在dist的后面8个bit放入更新使用的方向k
		newDist = __int_as_float(__float_as_int(newDist) & 0xFFFFFF00 | k);

		//oldDist是atomicMin()返回的值，返回的是此次原子修改前的值,无论是否成功
		int oldDist = atomicMin((int*)(d_updateDistPtr + neighborSmallIdx), __float_as_int(newDist));

		//如果修改成功了
		if (__int_as_float(oldDist) > newDist)
		{
			modified = 1;
		}
	}

	int warpOffset;

	if (modified)
	{
		//int pos = atomicAdd(&d_compact_size, 1);
		//d_frontier_compact_2[pos] = neighborSmallIdx;

		auto g = coalesced_threads();
		int warp_res;
		int rank = g.thread_rank();
		if (rank == 0)
			warp_res = atomicAdd(&d_compact_size, g.size());

		warp_res = g.shfl(warp_res, 0);
		int result = warp_res + rank;

		d_frontier_compact_2[result] = neighborSmallIdx;
	}
}


__global__
void ftTracingUpdateKernel(int* d_compress, int* d_decompress, int* d_frontier_compact, float* d_distPtr, float* d_updateDistPtr, unsigned char* parentSimplePtr_compact, unsigned char* d_activeMat_compact,
	int width, int height, int slice, int newSize, int compact_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid == 0)
		d_compact_size = 0;
	if (tid >= compact_size) return;

	//smallIdx: The index in the compressed image.
	//fullIdx: The index in the original image. 
	//These indices can be transformed to each other by "d_compress" or "d_decompress" array.
	int smallIdx = d_frontier_compact[tid];
	int fullIdx = d_decompress[smallIdx];


	//更新之前的direction
	//direction: One of the 26-way neighbor. This value stores the parent information of nodes. 
	int direction = parentSimplePtr_compact[smallIdx];
	//更新之前的seed(就是属于哪个种子)
	//In common, one element is extended from one seed.
	int z = fullIdx / (width * height);
	int y = fullIdx % (width * height) / width;
	int x = fullIdx % width;

	//更新之后(存储再updateDist中的) dist和direction
	float newDist = d_updateDistPtr[smallIdx];
	int directionUpdate = __float_as_int(newDist) & 0xFF;

	//更新之后的parent(根据directionUpdate计算)
	//The renewed parent
	int newParent;
	if (directionUpdate == 0xff)
		newParent = -1;
	else
		newParent = (z - dz3d26const[directionUpdate]) * width * height + (y - dy3d26const[directionUpdate]) * width + (x - dx3d26const[directionUpdate]);
	int newParentSmallIdx = d_compress[newParent];


	d_distPtr[smallIdx] = newDist;
	parentSimplePtr_compact[smallIdx] = directionUpdate;
	d_activeMat_compact[smallIdx] = ALIVE;
}


__global__
void copyDist2ImgKernel(unsigned char* d_imagePtr_compact, float* d_distPtr, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	float dist = d_distPtr[smallIdx];
	unsigned char result = 0;
	if (dist >=0 && dist <= 255)
		result = dist;
	d_imagePtr_compact[smallIdx] = result;
}

//featureTransform:输入原图，将结果
void featureTransForm(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_compress, int* d_decompress, int* d_parentPtr_compact, unsigned char* d_activeMat_compact, int width, int height, int slice, int newSize)
{
	cudaError_t errorCheck;

	//d_distPtr: 用于存储最短路的dist值
	//d_updateDistPtr: 用于存储即将更新的dist值(并行最短路中，需要先存储将要更新的dist，同步后再更新
	//Two arrays (one of this is temporary) to store the distance value in fast-marching 
	float* d_distPtr;
	float* d_updateDistPtr;


	cudaMalloc((void**)&d_distPtr, sizeof(float) * newSize);
	cudaMalloc((void**)&d_updateDistPtr, sizeof(float) * newSize);

	thrust::fill(thrust::device, d_distPtr, d_distPtr + newSize, 1e10f);
	thrust::fill(thrust::device, d_updateDistPtr, d_updateDistPtr + newSize, 1e10f);

	//d_parentSimplePtr: 为了让parent信息能和dist信息存储在同一个float32中，将parent信息压缩为一个字节(即，只存储一个方向)
	//The parent information are simplified to only one direction. The direction is from one of the 26-way neighbors.
	//As for the direction is only a 8-bit value, it can be integrated into the 32-bit floating distance value.
	unsigned char* d_parentSimplePtr;
	cudaMalloc(&d_parentSimplePtr, sizeof(unsigned char) * newSize);
	cudaMemset(d_parentSimplePtr, 0xff, sizeof(unsigned char) * newSize);

	//d_frontier_compact: 存储对锋面压缩后的结果，仅存储在锋面中的元素的下标
	//thrust::device_vector<int>dv_frontier_compact(newSize);
	int* d_frontier_compact;
	int* d_frontier_compact2;

	cudaMalloc(&d_frontier_compact, sizeof(int) * newSize);
	cudaMalloc(&d_frontier_compact2, sizeof(int) * newSize);
	cudaMemset(d_frontier_compact, 0, sizeof(int) * newSize);
	cudaMemset(d_frontier_compact2, 0, sizeof(int) * newSize);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before ft preprocesss " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	ftPreProcessKernel << < (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, d_distPtr, d_updateDistPtr, d_frontier_compact, d_activeMat_compact, width, height, slice, newSize);


	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During ft preprocess " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	int compact_size = newSize;

	int counter = 0;
	int blockNum = (newSize - 1) / 512 + 1;

	int* h_result;
	int* d_result;
	cudaHostAlloc(&h_result, sizeof(int), cudaHostRegisterMapped);
	*h_result = 0;
	cudaHostGetDevicePointer(&d_result, h_result, 0);

	int ping_pong = 0;
	int* f1;
	int* f2;

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before ft iteration " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	while (1)
	{
		//std::cerr << "Iter: " << counter << " frontier size: " << compact_size << std::endl;

		//Switch the memory location of current tracing frontier and the next tracing frontier
		if (ping_pong == 0)
		{
			f1 = d_frontier_compact; f2 = d_frontier_compact2;
		}
		else
		{
			f1 = d_frontier_compact2; f2 = d_frontier_compact;
		}

		ftTracingExtendKernel_warpShuffle_atomic << <(compact_size - 1) / 32 + 1, 1024 >> > (d_imagePtr, d_imagePtr_compact, f1, d_compress, d_decompress, d_distPtr, d_updateDistPtr,
			width, height, slice, newSize, compact_size, f2, d_activeMat_compact);


		cudaMemcpyFromSymbol((void*)&compact_size, d_compact_size, sizeof(int));

		if (compact_size == 0)
			break;

		ftTracingUpdateKernel << <(compact_size - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, f2, d_distPtr, d_updateDistPtr, d_parentSimplePtr, d_activeMat_compact,
			width, height, slice, newSize, compact_size);

		ping_pong = 1 - ping_pong;
		counter++;
	}

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During ft iteration " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}


	//初始追踪结束以后，将存储parent的方向改为存储完整的parent信息用于后面的处理
	//After the initial tracing, change the parent directions to the full parent indices for later processing.
	ftChangeSimpleParentToFull(d_compress, d_decompress, d_parentPtr_compact, d_parentSimplePtr, width, height, slice, newSize);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During changeSimpleParent " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	copyDist2ImgKernel << < (newSize - 1) / 256 + 1, 256 >> > (d_imagePtr_compact, d_distPtr, newSize);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During copyDist2Img " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	cudaFree(d_distPtr);
	cudaFree(d_updateDistPtr);
	cudaFree(d_parentSimplePtr);
	cudaFree(d_frontier_compact);
	cudaFree(d_frontier_compact2);
}


__global__ void changeParentKernel_compact(int* d_compress, int* d_decompress, int* d_parentPtr_compact, unsigned char* d_parentSimplePtr, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;

	int fullIdx = d_decompress[smallIdx];
	int direction;
	int offset = width * height * slice;
	int z = fullIdx / (width * height);
	int y = (fullIdx % (width * height)) / width;
	int x = fullIdx % width;

	int parentfullIdx, parentSmallIdx;

	direction = d_parentSimplePtr[smallIdx];
	if (direction != 0xff)
	{
		parentfullIdx  = (z - dz3d26const[direction]) * width * height + (y - dy3d26const[direction]) * width + (x - dx3d26const[direction]);
		parentSmallIdx = d_compress[parentfullIdx];
		d_parentPtr_compact[smallIdx] = parentSmallIdx;
	}
}


//初始追踪结束以后，将存储parent的方向改为存储完整的parent信息用于后面的处理
//After the initial tracing, change the parent directions to the full parent indices for later processing.
void ftChangeSimpleParentToFull(int* d_compress, int* d_decompress, int* d_parentPtr_compact, unsigned char* d_parentSimplePtr, int width, int height, int slice, int newSize)
{
	cudaError_t errorCheck;
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "Before changeParentKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}

	changeParentKernel_compact << <(newSize - 1) / 256 + 1, 256 >> > (d_compress, d_decompress, d_parentPtr_compact, d_parentSimplePtr, width, height, slice, newSize);

	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "In changeParentKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
}




__global__
void SetFtForBackgroundKernel(int* d_ftarr, int width, int height, int slice)
{
	int fullIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (fullIdx >= width * height * slice) return;
	d_ftarr[fullIdx] = fullIdx;
}

__global__
void SetFtForForegroundKernel(int* d_decompress, int* d_ftarr, int* d_parentPtr_compact, int width, int height, int slice, int newSize)
{
	int smallIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (smallIdx >= newSize) return;
	int parent = d_parentPtr_compact[smallIdx];
	if (parent == -1) return; //如果已经是起点了，退出
	while (d_parentPtr_compact[parent] != -1 && d_parentPtr_compact[parent] != parent)
	{
		parent = d_parentPtr_compact[parent]; //不断找parent直到起点
	}
	int fullIdx = d_decompress[smallIdx];
	int parentFullIdx = d_decompress[parent];
	d_ftarr[fullIdx] = parentFullIdx;
}

//通过访问parent信息找到每个点对应的最近背景点位置，即ft数组
void findFtPoints(int* d_decompress, int* d_ftarr, int* d_parentPtr_compact, int width, int height, int slice, int newSize)
{
	cudaError_t errorCheck;
	TimerClock timer;
	timer.update();
	
	SetFtForBackgroundKernel << < (width * height * slice - 1) / 256 + 1, 256 >> > (d_ftarr, width, height, slice);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During SetFtForBackgroundKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "SetFtForBackgroundKernel cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();
	
	SetFtForForegroundKernel << < (newSize - 1) / 256 + 1, 256 >> > (d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);
	cudaDeviceSynchronize();
	errorCheck = cudaGetLastError();
	if (errorCheck != cudaSuccess) {
		std::cerr << "During SetFtForForegroundKernel " << cudaGetErrorString(errorCheck) << std::endl;
		system("pause");
		return;
	}
	std::cerr << "SetFtForForegroundKernel cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

}

void findFtPointsHost(int* h_decompress, int* h_ftarr, int* h_parentPtr_compact, int width, int height, int slice, int newSize)
{
	for (int i = 0; i < width * height * slice; i++)
		h_ftarr[i] = i;

	std::cerr << "newSize: " << newSize << std::endl;
	for (int smallIdx = 0; smallIdx < newSize; smallIdx++)
	{
		if (smallIdx % 100000 == 0) std::cerr << "iter: " <<  smallIdx << std::endl;
		int parent = h_parentPtr_compact[smallIdx];
		//std::cerr << "original parent" << parent << std::endl;
		if (parent == -1) continue;
		while (h_parentPtr_compact[parent] != -1)
		{
			//std::cerr << parent << std::endl;
			parent = h_parentPtr_compact[parent]; //不断找parent直到起点
		}
		int fullIdx = h_decompress[smallIdx];
		int parentFullIdx = h_decompress[parent];
		h_ftarr[fullIdx] = parentFullIdx;
	}
}

//将ftArr存储的最远点位置再次转为距离
void convertFtPoints2Dist(int* h_ftarr, unsigned char* h_distPtr, int width, int height, int slice)
{
	for (int z = 0; z < slice; z++)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				int cur = z * width * height + y * width + x;
				int target = h_ftarr[cur];
				if (cur == target) h_distPtr[cur] = 0;
				int tz = target / (width * height);
				int ty = target % (width * height) / width;
				int tx = target % width;
				float dist = sqrtf((tz - z) * (tz - z) + (ty - y) * (ty - y) + (tx - x) * (tx - x));
				if (dist > 255) std::cerr << "dist out of range! " << dist << std::endl;
				h_distPtr[cur] = dist;
			}
}