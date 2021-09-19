#pragma once

#include <opencv2/highgui.hpp>
#include <iostream>
#include <Windows.h>

using namespace cv;
using namespace std;

Mat SuperxbrScaling(Mat image, int factor);

UMat SuperxbrScaling(UMat image, int factor);