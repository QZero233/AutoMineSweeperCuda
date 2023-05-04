#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "pch.h"

#include "data.h"

#include<stdio.h>

#include <jni.h>

//void freeImage(Image* image);
void solve(Image* game, jlong* targets, jdouble* result, int targetSize, Profile* profile);

__global__ void solveCuda(Image* game, jlong* targets, jdouble* result, int targetSize, Profile* profile);
//__global__ void freeImageCuda(Image* image);

