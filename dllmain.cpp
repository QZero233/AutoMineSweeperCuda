// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include "com_qzero_mine_recognize_NativeRecognizer.h"

#include "cudaWorker.cuh"

#include <iostream>

using namespace std;

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

JNIEXPORT jlong JNICALL Java_com_qzero_mine_recognize_NativeRecognizer_loadImage
(JNIEnv* env, jobject, jobject matrix) {
    //Parse matrix object
    jclass matrixClass = env->GetObjectClass(matrix);
    jfieldID rowFieldId = env->GetFieldID(matrixClass, "row", "I");
    jfieldID colFieldId = env->GetFieldID(matrixClass, "col", "I");
    jfieldID arrFieldId = env->GetFieldID(matrixClass, "arr", "[I");

    int row = env->GetIntField(matrix, rowFieldId);
    int col = env->GetIntField(matrix, colFieldId);
    jintArray image= (jintArray)env->GetObjectField(matrix, arrFieldId);

    //Copy data to cuda
    jsize len = env->GetArrayLength(image);
    jboolean isCopy = false;
    jint* imageArray = env->GetIntArrayElements(image, &isCopy);

    jint* cudaArray = NULL;
    cudaMalloc((void**)&cudaArray, len * sizeof(jint));

    cudaMemcpy((void*)cudaArray, (void*)imageArray, len * sizeof(jint), cudaMemcpyHostToDevice);

    env->ReleaseIntArrayElements(image, imageArray, JNI_ABORT);

    //Create struct and save to cuda
    Image* imageStruct = NULL;
    cudaMalloc((void**)&imageStruct, sizeof(Image));

    Image hostImage;
    hostImage.ptr = cudaArray;
    hostImage.rowNum = row;
    hostImage.colNum = col;

    cudaMemcpy((void*)imageStruct, (void*)&hostImage, sizeof(Image), cudaMemcpyHostToDevice);

    return (jlong)imageStruct;
}

JNIEXPORT jlong JNICALL Java_com_qzero_mine_recognize_NativeRecognizer_loadProfile
(JNIEnv* env, jobject, jintArray xDivides, jintArray yDivides) {
    jsize xDivideSize = env->GetArrayLength(xDivides);
    jsize yDivideSize = env->GetArrayLength(yDivides);

    //Copy array
    jboolean isCopy;
    jint* xDividesHost = env->GetIntArrayElements(xDivides, &isCopy);
    jint* yDividesHost = env->GetIntArrayElements(yDivides, &isCopy);

    jint* xDividesCuda;
    jint* yDividesCuda;
    cudaMalloc((void**)&xDividesCuda, xDivideSize*sizeof(jint));
    cudaMalloc((void**)&yDividesCuda, yDivideSize*sizeof(jint));

    cudaMemcpy(xDividesCuda, xDividesHost, xDivideSize * sizeof(jint), cudaMemcpyHostToDevice);
    cudaMemcpy(yDividesCuda, yDividesHost, yDivideSize * sizeof(jint), cudaMemcpyHostToDevice);

    //Format struct
    Profile profile;
    profile.xDivides = xDividesCuda;
    profile.yDivides = yDividesCuda;
    profile.xNum = xDivideSize - 1;
    profile.yNum = yDivideSize - 1;

    //Copy profile
    Profile* profileCuda;
    cudaMalloc((void**)&profileCuda, sizeof(Profile));
    cudaMemcpy(profileCuda, &profile, sizeof(Profile), cudaMemcpyHostToDevice);

    return (jlong)profileCuda;
}

JNIEXPORT void JNICALL Java_com_qzero_mine_recognize_NativeRecognizer_releaseImage
(JNIEnv*, jobject, jlong ptr) {
    Image* image = (Image*)ptr;
    //freeImage(image);
}

JNIEXPORT jintArray JNICALL Java_com_qzero_mine_recognize_NativeRecognizer_calculateLikelihood
(JNIEnv* env, jobject, jlong game, jlongArray targets, jlong profilePtr, jint xNum, jint yNum) {

    jsize targetSize = env->GetArrayLength(targets);

    jboolean isCopy = 0;
    jlong* targetPtrs = env->GetLongArrayElements(targets, &isCopy);
    Image* gamePtr = (Image*)game;

    //Malloc space for result
    int resultSize = xNum * yNum;
    jint* result;
    cudaMalloc((void**)&result, resultSize * sizeof(jint));

    
    solve(gamePtr, targetPtrs, result, targetSize, (Profile*)profilePtr, xNum, yNum);

    //Copy result to host
    jint* hostResult = new jint[resultSize];
    cudaMemcpy(hostResult, result, resultSize * sizeof(jint), cudaMemcpyDeviceToHost);

    //Format int array
    jintArray resultArray = env->NewIntArray(resultSize);
    jint* resultNative = env->GetIntArrayElements(resultArray, &isCopy);

    memcpy(resultNative, hostResult, resultSize * sizeof(jint));

    env->ReleaseIntArrayElements(resultArray, resultNative, 0);

    //Release resources
    cudaFree(result);
    delete[] hostResult;

    return resultArray;
}