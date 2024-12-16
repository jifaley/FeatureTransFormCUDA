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
函数：addGlobalThreshold
功能：给d_imagePtr 指向的图像添加全局阈值
*/
void addGlobalThreshold(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);

/*
函数：addDarkPadding
功能：给d_imagePtr 指向的图像进行补充
实现：对于足够亮的区域，将其周边的暗区灰度置为1
根据：试图填补不同亮区之间的缝隙，使得后面追踪时能成功连接相邻的亮区
*/
void addDarkPadding(unsigned char* d_imagePtr, int width, int height, int slice, unsigned char threshold);



/*
函数：getGlobalThreshold
功能：自适应得到体数据的背景截断阈值
*/
int getGlobalThreshold(unsigned char* h_imagePtr, unsigned char* d_imagePtr, int width, int height, int slice);