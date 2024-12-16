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
������compactImage
���ܣ�ѹ��ԭͼ��ȥ����0���֡�
�����d_imagePtr_compact(ѹ�����ͼ)��d_compress (ԭͼ->ѹ��ͼӳ��)��d_decompress(ѹ��ͼ->ԭͼӳ�䣩��newSize(ѹ�����ͼ��С)
ʵ�֣�ʹ��thrust���copy_if ���� remove_if ����
*/
void compactImage(unsigned char* d_imagePtr, unsigned char*& d_imagePtr_compact, int* &d_compress, int* &d_decompress, int width, int height, int slice, int& newSize);

/*
������recoverImage
���ܣ���ѹ����������ݷ��ص�ԭʼ����
*/
void recoverImage(unsigned char* d_imagePtr, unsigned char* d_imagePtr_compact, int* d_decompress, int newSize);

