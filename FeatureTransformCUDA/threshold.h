#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "utils.h"
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

/*
������addGlobalThreshold
���ܣ���d_imagePtr ָ���ͼ�����ȫ����ֵ
*/
void addGlobalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);

/*
������addLocalThreshold
���ܣ���d_imagePtr ָ���ͼ����Ӿֲ���ֵ
ʵ�֣����ȸ���blockSize������ͼ��ֿ飬�ֱ�ͳ�ƻҶ�ֱ��ͼ��ֻ����������������ǰ5%��ֵ��
���ݣ���Ԫ�ź�һ����Ա�����˵�������ġ�
ȱ�㣺�������Ϊ���Եķֿ�ЧӦ��Ӧ����Ӳ�ֵ������������
*/
void addLocalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, int blockSize, int globalThreshold);


/*
������addDarkPadding
���ܣ���d_imagePtr ָ���ͼ����в���
ʵ�֣������㹻�������򣬽����ܱߵİ����Ҷ���Ϊ1
���ݣ���ͼ���ͬ����֮��ķ�϶��ʹ�ú���׷��ʱ�ܳɹ��������ڵ�����
*/
void addDarkPadding(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);


int getGlobalThreshold(unsigned char* h_imagePtr, unsigned char* d_imagePtr, int width, int height, int slice);

void medianFilter(unsigned char* h_in, unsigned char* h_out, int width, int height, int slice);

void addGlobalThreshold_uppercut(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);