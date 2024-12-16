#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
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
������addDarkPadding
���ܣ���d_imagePtr ָ���ͼ����в���
ʵ�֣������㹻�������򣬽����ܱߵİ����Ҷ���Ϊ1
���ݣ���ͼ���ͬ����֮��ķ�϶��ʹ�ú���׷��ʱ�ܳɹ��������ڵ�����
*/
void addDarkPadding(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);



/*
������getGlobalThreshold
���ܣ�����Ӧ�õ������ݵı����ض���ֵ
*/
int getGlobalThreshold(unsigned char* h_imagePtr, unsigned char* d_imagePtr, int width, int height, int slice);