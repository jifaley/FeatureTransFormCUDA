#include "direction.h"
#include <cmath>
#include <string>
#include "neuron.h"

#define M_PI 3.1415926

__constant__ const int dx3d26const[26] = { -1,-1,-1,-1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1 };
__constant__ const int dy3d26const[26] = { -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1 };
__constant__ const int dz3d26const[26] = { -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1 };



void CalcDirection(unsigned char* h_imagePtr, int width, int height, int slice)
{
	unsigned char* d_imagePtr;
	cudaMalloc(&d_imagePtr, sizeof(unsigned char) * width * height * slice);
	cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyHostToDevice);


	std::string  soma_filename = "test_2_somas.swc";
	std::cerr << "soma file: " << soma_filename << std::endl;

	std::vector<NeuronNode> rootArr = readSWCFile(soma_filename, 1, 1, 1);

	int rootNum = rootArr.size();

	int* h_rootPos = (int*)malloc(sizeof(int) * rootNum);

	for (int i = 0; i < rootNum; i++)
	{
		int rootIdx = rootArr[i].z * width * height + rootArr[i].y * width + rootArr[i].x;
		h_rootPos[i] = rootIdx;

		int value = h_imagePtr[rootIdx];
		std::cerr << "rootId: " << i << " value: " << value << std::endl;
	}

	int* d_rootPos;
	cudaMalloc(&d_rootPos, sizeof(int) * rootNum);
	cudaMemcpy(d_rootPos, h_rootPos, sizeof(int) * rootNum, cudaMemcpyHostToDevice);


	Vec3f* d_resultDirections;
	cudaMalloc((void**)&d_resultDirections, sizeof(Vec3f) * rootNum);


	//这里调用主要计算函数。DirectionSphereCUDA是使用球坐标系计算主方向的,也是默认选项；函数内部可以更换不同的kernel
	calculateDominantDirectionSphereCUDA(d_imagePtr, width, height, slice, d_rootPos, d_resultDirections, rootNum);


	Vec3f* h_resultDirections = new Vec3f[rootNum];
	cudaMemcpy(h_resultDirections, d_resultDirections, sizeof(Vec3f) * rootNum, cudaMemcpyDeviceToHost);

	//for (int i = 0; i < seedCount; i++)
	//{
	//	std::cerr << "i: " << i << " mainDirection: " << h_resultDirections[i] << std::endl;
	//}

	//所有计算已经完毕，接下来是准备输出的SWC
	//SWC格式： id, color, x, y, z, radius, parentId (-1代表无parent)

	std::vector<NeuronNode> nodes;

	for (int i = 0; i < rootNum; i++)
	{
		int seedIdx = h_rootPos[i];
		int cz = seedIdx / (width * height);
		int cy = seedIdx % (width * height) / width;
		int cx = seedIdx % width;

		int curColor = (i) % 12 + 2;

		nodes.push_back(NeuronNode(nodes.size() + 1, curColor, cx, cy, cz, (16) / 4.0, -1));

		int rootIdx = nodes.size();

		Vec3f dir = h_resultDirections[i];

		//std::cerr << dir << std::endl;

		if (dir.magnitude() < 1e-5) continue;

		//下面完全是为了突出maindirection，在中心点的左右画线
		for (int j = 1; j <= 16; j++)
		{
			float nx = cx + j * dir.x;
			float ny = cy + j * dir.y;
			float nz = cz + j * dir.z;

			if (nz < 0 || nz >= slice || ny < 0 || ny >= height || nx < 0 || nx >= width)
				break;

			nodes.push_back(NeuronNode(nodes.size() + 1, curColor, nx, ny, nz, (16 - j) / 4.0, nodes.size()));
		}
		//画另一半
		for (int j = 1; j <= 16; j++)
		{
			float nx = cx - j * dir.x;
			float ny = cy - j * dir.y;
			float nz = cz - j * dir.z;

			if (nz < 0 || nz >= slice || ny < 0 || ny >= height || nx < 0 || nx >= width)
				break;

			if (j == 1)
				nodes.push_back(NeuronNode(nodes.size() + 1, curColor, nx, ny, nz, (16 - j) / 4.0, rootIdx));
			else
				nodes.push_back(NeuronNode(nodes.size() + 1, curColor, nx, ny, nz, (16 - j) / 4.0, nodes.size()));
		}
	}

	//下面是输出结果
	std::string output_file = "maindirection.swc";

	//需要y方向颠倒就使用这一句
	for (int i = 0; i < nodes.size(); i++)
		nodes[i].y = height - nodes[i].y - 1;

	std::cout << "Size(Nodes): " << nodes.size() << std::endl;
	// Convert the node list to SWC format string
	std::string swc = to_swc(nodes);

	// Write the SWC string to the output file
	std::ofstream ofs(output_file);
	if (ofs.is_open()) {
		ofs << swc;
		ofs.close();
		std::cout << "Successfully wrote the result to " << output_file << std::endl;
	}
	else {
		std::cerr << "Error: cannot open file " << output_file << std::endl;
		exit(1);
	}

	cudaFree(d_imagePtr);
	cudaFree(d_resultDirections);
	cudaFree(d_rootPos);

}



void generateAndCopyConstantDirectionVectors(Vec3f* d_directions) {
	std::vector<Vec3f> directions;

	// 分割theta和phi  
	int thetaSegments = 12;
	int phiSegments = 36;

	for (int i = 0; i < thetaSegments; ++i) {
		for (int j = 0; j < phiSegments; ++j) {
			double theta = (M_PI / (thetaSegments - 1)) * i;
			double phi = (M_PI / phiSegments) * j;

			double x = sin(theta) * cos(phi);
			double y = sin(theta) * sin(phi);
			double z = cos(theta);

			directions.push_back(Vec3f(x, y, z));
		}
	}

	cudaMemcpy(d_directions, &directions[0], 36 * 12 * sizeof(Vec3f), cudaMemcpyHostToDevice);
}


__global__ void calculateDominantDirectionSphereCUDAKernel(Vec3f* d_directions, unsigned char* d_imagePtr, int width, int height, int slice, int* points, Vec3f* resultDirections, int numPoints) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // 获取当前线程处理的点的索引  

	if (idx < numPoints) {
		int cx = points[idx] % width;
		int cy = points[idx] % (width * height) / width;
		int cz = points[idx] / (width * height);

		int maxAverageIntensity = 0;

		int windowsize = 8;
		//modified by jifaley 20240510
		windowsize = 16;
		Vec3f maxDirection(0, 0, 0);

		for (int i = 0; i < 9 * 12; ++i) {
			int totalIntensity = 0;
			int validVoxels = 0;

			Vec3f curDirection = d_directions[i];

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx + j * curDirection.x + 0.5f;
				int ny = cy + j * curDirection.y + 0.5f;
				int nz = cz + j * curDirection.z + 0.5f;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < slice) {
					int index = nz * width * height + ny * width + nx;
					totalIntensity += d_imagePtr[index];
					validVoxels++;
				}
				else {
					break;  // 达到图像边界，停止延伸  
				}
			}

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx - j * curDirection.x + 0.5f;
				int ny = cy - j * curDirection.y + 0.5f;
				int nz = cz - j * curDirection.z + 0.5f;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < slice) {
					int index = nz * width * height + ny * width + nx;
					totalIntensity += d_imagePtr[index];
					validVoxels++;
				}
				else {
					break;  // 达到图像边界，停止延伸  
				}
			}

			if (validVoxels > 0) {
				int averageIntensity = totalIntensity / validVoxels;
				if (averageIntensity > maxAverageIntensity) {
					maxAverageIntensity = averageIntensity;
					maxDirection = curDirection;
				}
			}
		}

		resultDirections[idx] = maxDirection;
	}
}


__global__ void calculateDominantDirectionSphereCUDAKernelWideWindow(Vec3f* d_directions, unsigned char* d_imagePtr, int width, int height, int slice, int* points, Vec3f* resultDirections, int numPoints) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // 获取当前线程处理的点的索引  

	if (idx < numPoints) {
		int cx = points[idx] % width;
		int cy = points[idx] % (width * height) / width;
		int cz = points[idx] / (width * height);

		int maxAverageIntensity = 0;

		int windowsize = 8;
		//modified by jifaley 20240510
		windowsize = 50;

		Vec3f maxDirection(0, 0, 0);

		for (int i = 0; i < 9 * 12; ++i) {
			int totalIntensity = 0;
			int validVoxels = 0;

			Vec3f curDirection = d_directions[i];

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx + j * curDirection.x + 0.5f;
				int ny = cy + j * curDirection.y + 0.5f;
				int nz = cz + j * curDirection.z + 0.5f;

				if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= slice)
					break;

				for (int sx = nx -1; sx <= nx +1; sx++)
					for (int sy = ny - 1; sy <= ny + 1; sy++)
						for (int sz = nz - 1; sz <= nz + 1; sz++)
						{
							if (sx >= 0 && sx < width && sy >= 0 && sy < height && sz >= 0 && sz < slice)
							{
								int index = sz * width * height + sy * width + sx;
								totalIntensity += d_imagePtr[index];
								validVoxels++;
							}
						}
			}

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx - j * curDirection.x + 0.5f;
				int ny = cy - j * curDirection.y + 0.5f;
				int nz = cz - j * curDirection.z + 0.5f;

				if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= slice)
					break;

				for (int sx = nx - 1; sx <= nx + 1; sx++)
					for (int sy = ny - 1; sy <= ny + 1; sy++)
						for (int sz = nz - 1; sz <= nz + 1; sz++)
						{
							if (sx >= 0 && sx < width && sy >= 0 && sy < height && sz >= 0 && sz < slice)
							{
								int index = sz * width * height + sy * width + sx;
								totalIntensity += d_imagePtr[index];
								validVoxels++;
							}
						}
			}

			if (validVoxels > 0) {
				int averageIntensity = totalIntensity / validVoxels;
				if (averageIntensity > maxAverageIntensity) {
					maxAverageIntensity = averageIntensity;
					maxDirection = curDirection;
				}
			}
		}

		resultDirections[idx] = maxDirection;
	}
}



__global__ void calculateDominantDirectionSphereCUDAKernelInfinityWindow(Vec3f* d_directions, unsigned char* d_imagePtr, int width, int height, int slice, int* points, Vec3f* resultDirections, int numPoints) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // 获取当前线程处理的点的索引  

	if (idx < numPoints) {
		int cx = points[idx] % width;
		int cy = points[idx] % (width * height) / width;
		int cz = points[idx] / (width * height);

		int maxAverageIntensity = 0;
		int maxValidVoxels = 0;
		int maxTotalIntensity = 0;


		Vec3f maxDirection(0, 0, 0);

		int windowsize = 50;

		for (int i = 0; i < 36 * 12; ++i) {
			int totalIntensity = 0;
			int validVoxels = 0;
			int blankcount = 0;

			Vec3f curDirection = d_directions[i];

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx + j * curDirection.x + 0.5f;
				int ny = cy + j * curDirection.y + 0.5f;
				int nz = cz + j * curDirection.z + 0.5f;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < slice) {
					int index = nz * width * height + ny * width + nx;
					int intensity = d_imagePtr[index];

					totalIntensity += intensity;
					validVoxels++;

					if (intensity == 0)
					{
						blankcount++;
						if (blankcount >= 3) break;
					}
				}
				else {
					break;  // 达到图像边界，停止延伸  
				}
			}

			blankcount = 0;

			for (int j = 1; j <= windowsize; ++j) {
				int nx = cx - j * curDirection.x + 0.5f;
				int ny = cy - j * curDirection.y + 0.5f;
				int nz = cz - j * curDirection.z + 0.5f;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < slice) {
					int index = nz * width * height + ny * width + nx;
					int intensity = d_imagePtr[index];

					totalIntensity += intensity;
					validVoxels++;

					if (intensity == 0)
					{
						blankcount++;
						if (blankcount >= 3) break;
					}
				}
				else {
					break;  // 达到图像边界，停止延伸  
				}
			}

			if (validVoxels > 0) {
				int averageIntensity = totalIntensity / validVoxels;
		/*		if (averageIntensity > maxAverageIntensity) {
					maxAverageIntensity = averageIntensity;
					maxDirection = curDirection;
				}*/

				//if (totalIntensity > maxTotalIntensity) {
				//	maxTotalIntensity = totalIntensity;
				//	maxDirection = curDirection;
				//}

				if (validVoxels > maxValidVoxels)
				{
					maxValidVoxels = validVoxels;
					maxDirection = curDirection;
				}

				//printf("validVoxels: %d\n", validVoxels);
			}
		}
		
		resultDirections[idx] = maxDirection;
	}
}


void calculateDominantDirectionSphereCUDA(unsigned char* d_imagePtr, int width, int height, int slice, int* d_points, Vec3f* d_resultDirections, int numPoints)
{
	Vec3f* d_directions;
	cudaMalloc(&d_directions, sizeof(Vec3f) * 36 * 12);
	generateAndCopyConstantDirectionVectors(d_directions);
	int blockSize = 256;
	int numBlocks = (numPoints + blockSize - 1) / blockSize;

	//这里也可以换几种不同的kernel
	//calculateDominantDirectionSphereCUDAKernel << <numBlocks, blockSize >> > (d_directions, d_imagePtr, width, height, slice, d_points, d_resultDirections, numPoints);
	//calculateDominantDirectionSphereCUDAKernelWideWindow << <numBlocks, blockSize >> > (d_directions, d_imagePtr, width, height, slice, d_points, d_resultDirections, numPoints);
	calculateDominantDirectionSphereCUDAKernelInfinityWindow << <numBlocks, blockSize >> > (d_directions, d_imagePtr, width, height, slice, d_points, d_resultDirections, numPoints);
	cudaFree(d_directions);
}



//int main() {
//	// 示例用法  
//	// ... 其他初始化 ...  
//
//	// 分配GPU内存  
//	unsigned char* d_imagePtr;
//	int* d_points;
//	Vec3d* d_resultDirections;
//	cudaMalloc((void**)&d_imagePtr, sizeof(unsigned char) * width * height * slice);
//	cudaMalloc((void**)&d_points, sizeof(int) * numPoints);
//	cudaMalloc((void**)&d_resultDirections, sizeof(Vec3d) * numPoints);
//
//	// 将数据从主机内存复制到GPU内存  
//	cudaMemcpy(d_imagePtr, h_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_points, points, sizeof(int) * numPoints, cudaMemcpyHostToDevice);
//
//	// 调用CUDA函数  
//	int blockSize = 256;
//	int numBlocks = (numPoints + blockSize - 1) / blockSize;
//	calculateDominantDirectionCUDAKernel << <numBlocks, blockSize >> > (d_imagePtr, width, height, slice, d_points, d_resultDirections, numPoints);
//
//	// 将结果从GPU内存复制回主机内存  
//	Vec3d* h_resultDirections = new Vec3d[numPoints];
//	cudaMemcpy(h_resultDirections, d_resultDirections, sizeof(Vec3d) * numPoints, cudaMemcpyDeviceToHost);
//
//	// ... 对结果进行处理 ...  
//
//	// 释放GPU内存  
//	cudaFree(d_imagePtr);
//	cudaFree(d_points);
//	cudaFree(d_resultDirections);
//	delete[] h_resultDirections;
//
//	return 0;
//}