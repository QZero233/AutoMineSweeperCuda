#pragma once
#ifndef DATA_H
#define DATA_H

#include <jni.h>

struct Image {
	jint* ptr = NULL;
	int rowNum = 0;
	int colNum = 0;
};

struct Coordinate {
	int x;
	int y;
};

struct Profile {
	jint* xDivides;
	jint* yDivides;

	int xNum;
	int yNum;
};

#endif