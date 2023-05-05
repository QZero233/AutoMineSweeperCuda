#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "pch.h"

#include "data.h"

#include<iostream>

#include <jni.h>

#include <Windows.h>

using namespace std;

//void freeImage(Image* image);
void solve(Image* game, jlong* targets, jint* result, int targetSize, Profile* profile,
	int xNum, int yNum);

__global__ void solveCuda(Image* game, jlong* targets, jint* result, int targetSize, Profile* profile);
//__global__ void freeImageCuda(Image* image);


