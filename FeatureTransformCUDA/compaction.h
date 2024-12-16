#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/extrema.h>


/*
函数：compactImage
功能：压缩原图，去除非0部分。
输出：d_imagePtr_compact(压缩后的图)，d_compress (原图->压缩图映射)，d_decompress(压缩图->原图映射），newSize(压缩后的图大小)
实现：使用thrust库的copy_if 或者 remove_if 操作
*/
void compactImage(unsigned char* d_imagePtr, unsigned char*& d_imagePtr_compact, int* &d_compress, int* &d_decompress, int width, int height, int slice, int& newSize);

/*
函数：recoverImage
功能：将压缩数组的内容返回到原始数组
*/
void recoverImage(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize);

