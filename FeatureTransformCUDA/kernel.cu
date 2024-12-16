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

	
	//���ȫ����ֵ
	addGlobalThreshold(d_imagePtr, width, height, slice, globalThreshold);

	std::cerr << "set globalThreshold cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//��ԭͼ�н���ǰ����ı����㸳ֵΪ1����Ϊ֮�����չ���
	addDarkPadding(d_imagePtr, width, height, slice, globalThreshold);
	std::cerr << "add darkpadding cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	int* d_compress; //ѹ��ӳ�����
	int* d_decompress; //��ѹ��ӳ�����
	unsigned char* d_imagePtr_compact; //ѹ����ԭͼ
	int newSize; //ѹ���������ܴ�С
	
	compactImage(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);

	std::cerr << "OldSize: " << width * height * slice << " NewSize: " << newSize << std::endl;
	printf("Compress Ratio: %.2lf%%\n", 100.0 * newSize / (1.0 * width * height * slice));

	cudaDeviceSynchronize();
	std::cerr << "compaction cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();


	////////////////////����Ϊ����


	
	//����1: ֱ�ӽ�ԭͼ�任Ϊ����ֵ������addGreyWeightTransform() ���� ��d_imagePtr_compact ��Ϊ����任֮�������
	
	//addGreyWeightTransform(d_imagePtr, d_imagePtr_compact, d_compress, d_decompress, width, height, slice, newSize);
	//cudaDeviceSynchronize();
	//std::cerr << "GreyWeight Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	//timer.update();

	//����recoverImage() ��d_imagePtr_compact�е����ݽ�ѹ����ԭͼd_imagePtr��
	//recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);



	/////////////����1���


	///////////����2���ڱ任����ֵ�Ĺ�����ͬʱ�洢��չ�ĸ�����Ϣ��ֱ���ҵ�ÿ�����������ı���������洢��ft������
	
	int* d_parentPtr_compact;
	unsigned char* d_activeMat_compact;
	cudaMalloc(&d_parentPtr_compact, sizeof(int) * newSize * 2);
	cudaMalloc(&d_activeMat_compact, sizeof(unsigned char) * newSize);
	cudaMemset(d_parentPtr_compact, -1, sizeof(int) * newSize);
	cudaMemset(d_activeMat_compact, FARAWAY, sizeof(unsigned char) * newSize);

	
	//���룺ԭͼd_imagePtr_compact����ԭͼ�޸�Ϊ����ֵ�����Ҷ������d_parentPtr_compact������չ�ĸ��׽ڵ���Ϣ
	featureTransForm(d_imagePtr_compact, d_compress, d_decompress, d_parentPtr_compact, d_activeMat_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "Feature Transform cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	//����recoverImage() ��d_imagePtr_compact�е����ݽ�ѹ����ԭͼd_imagePtr��
	recoverImage(d_imagePtr, d_imagePtr_compact, d_decompress, newSize);

	cudaDeviceSynchronize();
	std::cerr << "RecoverImage cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();
	
	
	int* d_ftarr;
	cudaMalloc(&d_ftarr, sizeof(int) * width * height * slice);

	//ftarr���鼴ΪIMA3D Ҫ��ģ���������������λ�á�����֮ǰ�����parent��Ϣ���Ÿ��ټ���õ�
	findFtPoints(d_decompress, d_ftarr, d_parentPtr_compact, width, height, slice, newSize);

	cudaDeviceSynchronize();
	std::cerr << "findFtPoints cost " << timer.getTimerMilliSec() << "ms" << std::endl << std::endl;
	timer.update();

	int* h_ftarr = (int*)malloc(sizeof(int) * width * height * slice);
	cudaMemcpy(h_ftarr, d_ftarr, sizeof(int) * width * height * slice, cudaMemcpyDeviceToHost);


	////��֤��ʹ��ftarr�е�λ�����¼������ֵ���洢���ؽ�����h_distPtr��
	unsigned char* h_recDistPtr = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);
	convertFtPoints2Dist(h_ftarr, h_recDistPtr, width, height, slice);
	
	//����Ҫ���Ե���host�汾��findFtPoint
	//findFtPointsHost(h_decompress, h_ftarr, h_parentPtr_compact, width, height, slice, newSize);
	//free(h_parentPtr_compact);
	//free(h_decompress);
	
	//output.tif�洢featureTransForm()�õ��ľ���ͼ
	std::string outputFile = "output.tif";

	//rec_output.tif�洢ʹ��ft�����ؽ��õ��ľ���ͼ
	std::string reconstructedOutputFile = "rec_output.tif";

	cudaMemcpy(h_imagePtr, d_imagePtr, sizeof(unsigned char) * width * height * slice, cudaMemcpyDeviceToHost);

	saveTiff(outputFile.c_str(), h_imagePtr, size);

	saveTiff(reconstructedOutputFile.c_str(), h_recDistPtr, size);

	////////����2���

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
