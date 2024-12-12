#include "loadTiff.h"


unsigned char* loadImage(const std::string inputName, int* imageShape)
{

	TIFF *tif = TIFFOpen(inputName.c_str(), "r");
	if (tif == nullptr)
	{
		std::cerr << "读入图像路径错误,请重新确认";
		return nullptr;
	}

	int width, height;
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

	int nTotalFrame = TIFFNumberOfDirectories(tif);
	int slice = nTotalFrame;

	std::cerr << "width: " << width << std::endl;
	std::cerr << "height: " << height << std::endl;
	std::cerr << "Slice: " << nTotalFrame << std::endl;


	unsigned char* buffer = (unsigned char*)malloc(sizeof(unsigned char) * width * height * slice);

	TIFFSetDirectory(tif, 0);

	for (int s = 0; s < nTotalFrame; s++) {
		for (int i = 0; i < height; i++) {
			TIFFReadScanline(tif, buffer + s * width * height + (height - 1 - i) * width, i);
		}
		TIFFReadDirectory(tif);  // Move to the next frame  
	}

	TIFFClose(tif);

	imageShape[0] = width;
	imageShape[1] = height;
	imageShape[2] = slice;


	_ASSERT(buffer != NULL);
	return buffer;
}



void saveTiff(const char *path, unsigned char *buffer, int *size)
{
	int width = size[0];
	int height = size[1];
	int slice = size[2];

	TIFF* out = TIFFOpen(path, "w");
	if (out)
	{
		int N_size = 0;
		size_t nCur = 0;
		//unsigned char den = (sizeof(T) == 1) ? 1 : 4;
		do {
			TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
			TIFFSetField(out, TIFFTAG_PAGENUMBER, slice);
			TIFFSetField(out, TIFFTAG_IMAGEWIDTH, (uint32)width);
			TIFFSetField(out, TIFFTAG_IMAGELENGTH, (uint32)height);
			//TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, 2);
			/*TIFFSetField(out, TIFFTAG_YRESOLUTION, 196.0f);
			TIFFSetField(out, TIFFTAG_XRESOLUTION, 204.0f);*/
			TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
			//TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);
			// 
			TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    //根据图像位深填不同的值
			TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
			TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
			TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
			TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
			TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, height);


			for (int m = 0; m < height; m++)
			{
				TIFFWriteScanline(out, &buffer[N_size + width * m], m, 0);
			}
			//TIFFWriteEncodedStrip(out, 0, &buffer[N_size], width * height);      //另一种写入方法

			++nCur;
			N_size = N_size + width * height;
		} while (TIFFWriteDirectory(out) && nCur < slice);
		TIFFClose(out);

		std::cout << "save over" << std::endl;
	}
}

