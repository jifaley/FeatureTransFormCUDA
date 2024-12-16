#pragma once
#include "libtiff/include/tiffio.h"
#include <iostream>
#include <string>


void saveTiff(const char *path, unsigned char *buffer, int *size);

unsigned char* loadImage(const std::string inputName, int* imageShape);