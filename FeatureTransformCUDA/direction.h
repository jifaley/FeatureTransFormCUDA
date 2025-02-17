#pragma once
#include "utils.h"
#include <cuda_runtime.h>  
#include <device_launch_parameters.h>  

class Vec3d {
public:
	int x, y, z;
	__host__ __device__ Vec3d() : x(0), y(0), z(0) { }
	__host__ __device__ Vec3d(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}

	friend std::ostream& operator<<(std::ostream& os, const Vec3d& vec) {
		os << "Vec3d(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
		return os;
	}
};

class Vec3f {
public:
	float x, y, z;
	__host__ __device__ Vec3f() : x(0), y(0), z(0) { }
	__host__ __device__ Vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vec3f(Vec3d rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}
	__host__ __device__ float dot(const Vec3f& other) const 
	{
		return x * other.x + y * other.y + z * other.z;
	}

	friend std::ostream& operator<<(std::ostream& os, const Vec3f& vec) {
		os << "Vec3f(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
		return os;
	}

	__host__ __device__ float magnitude() const {
		return std::sqrt(x * x + y * y + z * z);
	}
};


//在主程序里调用此函数即可
void CalcDirection(unsigned char* h_imagePtr, int width, int height, int slice);

void calculateDominantDirectionSphereCUDA(unsigned char* d_imagePtr, int width, int height, int slice, int* d_points, Vec3f* d_resultDirections, int numPoints);