#include <opencv2/highgui.hpp>
#include <iostream>
#include <Windows.h>

#include "algorithm.h"

using namespace cv;
using namespace std;

int main()
{
	string filepath = "fish.png";
	Mat src = imread(filepath);

	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);

	Mat result = SuperxbrScaling(src, 2);
	
	QueryPerformanceCounter(&nEndTime);

	double time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;
	cout << "Time：" << time * 1000 << " ms." << endl;

	//namedWindow("src");
	//imshow("src", src);
	//namedWindow("result");
	//imshow("result", result);
	//waitKey(0);
	return 0;
}