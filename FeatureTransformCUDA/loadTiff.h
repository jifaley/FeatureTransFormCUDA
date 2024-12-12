#pragma once
#include <tiffio.h>
#include <iostream>
#include <string>


void saveTiff(const char *path, unsigned char *buffer, int *size);

unsigned char* loadImage(const std::string inputName, int* imageShape);

void WriteIterToTIF(unsigned char* img, int width, int height, int slice, int iter);