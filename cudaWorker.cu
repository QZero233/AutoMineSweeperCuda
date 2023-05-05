#include "cudaWorker.cuh"
#include<algorithm>

__device__ int inline getIdx(int x, int y, Image* img) {
	return y * img->colNum + x;
}

//__global__ void freeImageCuda(Image* image) {
//	jint* ptr = image->ptr;
//	cudaFree(ptr);
//	cudaFree(image);
//}

__global__ void solveCuda(Image* game, jlong* targets, jint* result, int targetSize, Profile* profile) {

	extern __shared__ double resultLikelihood[];

	int targetX = blockIdx.x;
	int targetY = blockIdx.y;

	if (targetX >= profile->xNum || targetY >= profile->yNum)
		return;

	int targetOffset = targetSize * (targetY * profile->xNum + targetX);

	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < targetSize; i += stride) {
		//Calculate target i for (x,y)
		Image* target = (Image*)targets[i];

		Coordinate leftTop;
		leftTop.x = profile->xDivides[targetX];
		leftTop.y = profile->yDivides[targetY];

		Coordinate rightBottom;
		rightBottom.x = profile->xDivides[targetX + 1];
		rightBottom.y = profile->yDivides[targetY + 1];

		int xSize = rightBottom.x - leftTop.x + 1;
		int ySize = rightBottom.y - leftTop.y + 1;

		int xMin = min(xSize, target->colNum);
		int yMin = min(ySize, target->rowNum);

		double currentSum = 0;
		for (int x = xMin / 8; x < xMin * 7 / 8; x++) {
			for (int y = yMin / 8; y < yMin * 7 / 8; y++) {
				int srcColor = game->ptr[getIdx(leftTop.x + x, leftTop.y + y, game)];
				int targetColor = target->ptr[getIdx(x, y, target)];

				currentSum += std::abs(targetColor - srcColor);
			}
		}

		if (i == 0) {
			result[targetY * profile->xNum + targetX] = currentSum;
		}

		currentSum /= xMin * yMin;
		resultLikelihood[i] = currentSum;
		
	}

	__syncthreads();

	//Judge
	if (index == 0) {
		int bestIndex = -1;
		double bestLikelihood = 1e20;
		for (int j = 0; j < targetSize; j++) {
			if (resultLikelihood[j] < bestLikelihood) {
				bestLikelihood = resultLikelihood[j];
				bestIndex = j;
			}
		}

		result[targetY * profile->xNum + targetX] = bestIndex;
	}

}

void solve(Image* game, jlong* targets, jint* result, int targetSize, Profile* profile,
	int xNum, int yNum) {

	//DWORD start = GetTickCount();

	//Create temp array
	//double* resultLikelihood;
	//cudaMalloc((void**)&resultLikelihood, xNum * yNum * targetSize * sizeof(double));

	//Copy targets array
	jlong* cudaTargets = NULL;
	cudaMalloc((void**)&cudaTargets, targetSize * sizeof(jlong*));
	cudaMemcpy(cudaTargets, targets, targetSize * sizeof(jlong*), cudaMemcpyHostToDevice);

	//cudaDeviceSynchronize();
	
	//Invoke kernel
	dim3 blockSize((targetSize / 32 + 1) * 32);
	dim3 gridSize(128, 128);
	solveCuda << <gridSize, blockSize, targetSize*sizeof(double) >> > (game, cudaTargets, result, targetSize, profile);

	//Debug
	//cudaDeviceSynchronize();
	//cout << "After cuda solve " << (GetTickCount() - start) << endl;
	//double* d = new double[xNum * yNum * targetSize];
	//cudaMemcpy(d, resultLikelihood, xNum * yNum * targetSize * sizeof(double), cudaMemcpyDeviceToHost);
	//int cnt = 0;
	//for (int i = 0; i < xNum * yNum * targetSize; i++) {
	//	if (d[i] == 0)
	//		//cout << d[i] << endl;
	//		cnt++;
	//}
	//cout << cnt << endl;

	//Release
	cudaFree(cudaTargets);
	//cudaFree(resultLikelihood);
}

//void freeImage(Image* image) {
//	dim3 blockSize(1);
//	dim3 gridSize(1);
//	//freeImageCuda<<<gridSize, blockSize>>>(image);
//}