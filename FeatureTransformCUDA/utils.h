#pragma once
#include <vector>
#include <iostream>

//Some utils or definations

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


enum
{
	FARAWAY, TRIAL, ALIVE, DARKLEAF_PRUNED
};


