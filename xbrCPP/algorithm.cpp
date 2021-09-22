#include "algorithm.h"

using namespace cv;
using namespace std;

double df(double a, double b)
{
	double value = fabs(a - b);
	return value;
}

int getMin(int a, int b, int c, int d)
{
	int minVal = a;
	if (b < a) minVal = b;
	if (c < minVal) minVal = c;
	if (d < minVal) minVal = d;
	return minVal;
}

int getMax(int a, int b, int c, int d)
{
	int maxVal = a;
	if (b > a) maxVal = a;
	if (c > maxVal) maxVal = c;
	if (d > maxVal) maxVal = d;
	return maxVal;
}

double diagonal_edge(double y[4][4], double *wp)
{
	double dw1 = wp[0] * (df(y[0][2], y[1][1]) + df(y[1][1], y[2][0]) + df(y[1][3], y[2][2]) + df(y[2][2], y[3][1])) +
		wp[1] * (df(y[0][3], y[1][2]) + df(y[2][1], y[3][0])) +
		wp[2] * (df(y[0][3], y[2][1]) + df(y[1][2], y[3][0])) +
		wp[3] * df(y[1][2], y[2][1]) +
		wp[4] * (df(y[0][2], y[2][0]) + df(y[1][3], y[3][1])) +
		wp[5] * (df(y[0][1], y[1][0]) + df(y[2][3], y[3][2]));
	double dw2 = wp[0] * (df(y[0][1], y[1][2]) + df(y[1][2], y[2][3]) + df(y[1][0], y[2][1]) + df(y[2][1], y[3][2])) +
		wp[1] * (df(y[0][0], y[1][1]) + df(y[2][2], y[3][3])) +
		wp[2] * (df(y[0][0], y[2][2]) + df(y[1][1], y[3][3])) +
		wp[3] * df(y[1][1], y[2][2]) +
		wp[4] * (df(y[1][0], y[3][2]) + df(y[0][1], y[2][3])) +
		wp[5] * (df(y[0][2], y[1][3]) + df(y[2][0], y[3][1]));
	return dw1 - dw2;

}

Mat SuperxbrScaling(Mat image, int factor)
{
#pragma region Preprocessing
	int smallW = image.cols;
	int smallH = image.rows;
	int bigW = smallW * factor;
	int bigH = smallH * factor;

	Vec3b value;
	Mat result = Mat(bigH, bigW, CV_8UC3);

	uchar *srcData = (uchar*)image.data;
	uchar *dstData = (uchar*)result.data;

	// weight
	double wgt1 = 0.129633;
	double wgt2 = 0.129633;
	double w1 = -wgt1;
	double w2 = wgt1 + 0.5;
	double w3 = -wgt2;
	double w4 = wgt2 + 0.5;

	// initialization
	int r[4][4] = { 0 };
	int g[4][4] = { 0 };
	int b[4][4] = { 0 };
	int a[4][4] = { 0 };
	double Y[4][4] = { 0 };
	int *rPtr = *r;
	int *gPtr = *g;
	int *bPtr = *b;
	int *aPtr = *a;
	double *yPtr = *Y;

	double rf, gf, bf, af;
	int ri, gi, bi, ai;
	double d_edge;
	double min_r_sample, max_r_sample;
	double min_g_sample, max_g_sample;
	double min_b_sample, max_b_sample;
	double min_a_sample, max_a_sample;

	double rWeight[256] = { 0 };
	double gWeight[256] = { 0 };
	double bWeight[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		rWeight[i] = 0.2126 * i;
		gWeight[i] = 0.7152 * i;
		bWeight[i] = 0.0722 * i;
	}
#pragma endregion

#pragma region First Pass
	double wp[] = { 2.0, 1.0, -1.0, 4.0, -1.0, 1.0 };

	for (int y = 0; y < bigH; ++y)
	{
		for (int x = 0; x < bigW; ++x)
		{
			// central pixels on original images
			int cx = x / factor;
			int cy = y / factor;

			// sample supporting pixels in original image
			for (int sx = -1; sx <= 2; ++sx)
			{
				for (int sy = -1; sy <= 2; ++sy)
				{
					// clamp pixel locations
					int csy = sy + cy;
					int csx = sx + cx;
					csy = (csy < 0) ? 0 : csy;
					csy = (csy > smallH - 1) ? smallH - 1 : csy;
					csx = (csx < 0) ? 0 : csx;
					csx = (csx > smallW - 1) ? smallW - 1 : csx;

					// sample & add weighted components
					// 3 is channel number
					uchar* bSample = srcData + (3 * (smallW * csy + csx));
					uchar* gSample = bSample + 1;
					uchar* rSample = bSample + 2;
					uchar aSample = 255;

					int offset = 4 * (sx + 1) + (sy + 1);
					*(rPtr + offset) = *rSample;
					*(gPtr + offset) = *gSample;
					*(bPtr + offset) = *bSample;
					*(aPtr + offset) = aSample;
					*(yPtr + offset) = rWeight[*rSample] + gWeight[*gSample] + bWeight[*bSample];
				}
			}

			min_r_sample = getMin(r[1][1], r[2][1], r[1][2], r[2][2]);
			min_g_sample = getMin(g[1][1], g[2][1], g[1][2], g[2][2]);
			min_b_sample = getMin(b[1][1], b[2][1], b[1][2], b[2][2]);
			min_a_sample = getMin(a[1][1], a[2][1], a[1][2], a[2][2]);
			max_r_sample = getMax(r[1][1], r[2][1], r[1][2], r[2][2]);
			max_g_sample = getMax(g[1][1], g[2][1], g[1][2], g[2][2]);
			max_b_sample = getMax(b[1][1], b[2][1], b[1][2], b[2][2]);
			max_a_sample = getMax(a[1][1], a[2][1], a[1][2], a[2][2]);
			d_edge = diagonal_edge(Y, wp);

			if (d_edge <= 0)
			{
				rf = w1 * (r[0][3] + r[3][0]) + w2 * (r[1][2] + r[2][1]);
				gf = w1 * (g[0][3] + g[3][0]) + w2 * (g[1][2] + g[2][1]);
				bf = w1 * (b[0][3] + b[3][0]) + w2 * (b[1][2] + b[2][1]);
				af = w1 * (a[0][3] + a[3][0]) + w2 * (a[1][2] + a[2][1]);
			}
			else
			{
				rf = w1 * (r[0][0] + r[3][3]) + w2 * (r[1][1] + r[2][2]);
				gf = w1 * (g[0][0] + g[3][3]) + w2 * (g[1][1] + g[2][2]);
				bf = w1 * (b[0][0] + b[3][3]) + w2 * (b[1][1] + b[2][2]);
				af = w1 * (a[0][0] + a[3][3]) + w2 * (a[1][1] + a[2][2]);
			}
			// anti-ringing, clamp.
			rf = (rf < min_r_sample) ? min_r_sample : rf;
			rf = (rf > max_r_sample) ? max_r_sample : rf;
			gf = (gf < min_g_sample) ? min_g_sample : gf;
			gf = (gf > max_g_sample) ? max_g_sample : gf;
			bf = (bf < min_b_sample) ? min_b_sample : bf;
			bf = (bf > max_b_sample) ? max_b_sample : bf;
			af = (af < min_a_sample) ? min_a_sample : af;
			af = (af > max_a_sample) ? max_a_sample : af;

			ri = (rf - (int)rf == 0) ? rf : (rf + 1);
			gi = (gf - (int)gf == 0) ? gf : (gf + 1);
			bi = (bf - (int)bf == 0) ? bf : (bf + 1);
			ai = (af - (int)af == 0) ? af : (af + 1);

			ri = (ri < 0) ? 0 : ri;
			ri = (ri > 255) ? 255 : ri;
			gi = (gi < 0) ? 0 : gi;
			gi = (gi > 255) ? 255 : gi;
			bi = (bi < 0) ? 0 : bi;
			bi = (bi > 255) ? 255 : bi;
			ai = (ai < 0) ? 0 : ai;
			ai = (ai > 255) ? 255 : ai;

			uchar* bValue = srcData + (3 * (smallW * cy + cx));
			uchar* gValue = bValue + 1;
			uchar* rValue = bValue + 2;
			// (y, x)
			*(dstData + (3 * (bigW * y + x))) = *bValue;
			*(dstData + (3 * (bigW * y + x)) + 1) = *gValue;
			*(dstData + (3 * (bigW * y + x)) + 2) = *rValue;
			// (y, x+1)
			*(dstData + (3 * (bigW * y + x + 1))) = *bValue;
			*(dstData + (3 * (bigW * y + x + 1)) + 1) = *gValue;
			*(dstData + (3 * (bigW * y + x + 1)) + 2) = *rValue;
			// (y+1, x)
			*(dstData + (3 * (bigW * (y + 1) + x))) = *bValue;
			*(dstData + (3 * (bigW * (y + 1) + x)) + 1) = *gValue;
			*(dstData + (3 * (bigW * (y + 1) + x)) + 2) = *rValue;
			// (y+1, x+1)
			*(dstData + (3 * (bigW * (y + 1) + x + 1))) = bi;
			*(dstData + (3 * (bigW * (y + 1) + x + 1)) + 1) = gi;
			*(dstData + (3 * (bigW * (y + 1) + x + 1)) + 2) = ri;

			++x;
		}
		++y;
	}
#pragma endregion

#pragma region Second Pass
	wp[0] = 2.0;
	wp[1] = 0.0;
	wp[2] = 0.0;
	wp[3] = 0.0;
	wp[4] = 0.0;
	wp[5] = 0.0;

	for (int y = 0; y < bigH; ++y)
	{
		for (int x = 0; x < bigW; ++x)
		{
			// sample supporting pixels in original image
			for (int sx = -1; sx <= 2; ++sx)
			{
				for (int sy = -1; sy <= 2; ++sy)
				{
					// clamp pixel locations
					int csy = sx - sy + y;
					int csx = sx + sy + x;
					csy = (csy < 0) ? 0 : csy;
					csy = (csy > bigH - 1) ? bigH - 1 : csy;
					csx = (csx < 0) ? 0 : csx;
					csx = (csx > bigW - 1) ? bigW - 1 : csx;

					// sample & add weighted components
					// 3 is channel number
					uchar* bSample = dstData + (3 * (bigW * csy + csx));
					uchar* gSample = bSample + 1;
					uchar* rSample = bSample + 2;
					uchar aSample = 255;

					int offset = 4 * (sx + 1) + (sy + 1);
					*(rPtr + offset) = *rSample;
					*(gPtr + offset) = *gSample;
					*(bPtr + offset) = *bSample;
					*(aPtr + offset) = aSample;
					*(yPtr + offset) = rWeight[*rSample] + gWeight[*gSample] + bWeight[*bSample];
				}
			}

			min_r_sample = getMin(r[1][1], r[2][1], r[1][2], r[2][2]);
			min_g_sample = getMin(g[1][1], g[2][1], g[1][2], g[2][2]);
			min_b_sample = getMin(b[1][1], b[2][1], b[1][2], b[2][2]);
			min_a_sample = getMin(a[1][1], a[2][1], a[1][2], a[2][2]);
			max_r_sample = getMax(r[1][1], r[2][1], r[1][2], r[2][2]);
			max_g_sample = getMax(g[1][1], g[2][1], g[1][2], g[2][2]);
			max_b_sample = getMax(b[1][1], b[2][1], b[1][2], b[2][2]);
			max_a_sample = getMax(a[1][1], a[2][1], a[1][2], a[2][2]);

			d_edge = diagonal_edge(Y, wp);
			if (d_edge <= 0)
			{
				rf = w3 * (r[0][3] + r[3][0]) + w4 * (r[1][2] + r[2][1]);
				gf = w3 * (g[0][3] + g[3][0]) + w4 * (g[1][2] + g[2][1]);
				bf = w3 * (b[0][3] + b[3][0]) + w4 * (b[1][2] + b[2][1]);
				af = w3 * (a[0][3] + a[3][0]) + w4 * (a[1][2] + a[2][1]);
			}
			else
			{
				rf = w3 * (r[0][0] + r[3][3]) + w4 * (r[1][1] + r[2][2]);
				gf = w3 * (g[0][0] + g[3][3]) + w4 * (g[1][1] + g[2][2]);
				bf = w3 * (b[0][0] + b[3][3]) + w4 * (b[1][1] + b[2][2]);
				af = w3 * (a[0][0] + a[3][3]) + w4 * (a[1][1] + a[2][2]);
			}

			// anti-ringing, clamp.
			rf = (rf < min_r_sample) ? min_r_sample : rf;
			rf = (rf > max_r_sample) ? max_r_sample : rf;
			gf = (gf < min_g_sample) ? min_g_sample : gf;
			gf = (gf > max_g_sample) ? max_g_sample : gf;
			bf = (bf < min_b_sample) ? min_b_sample : bf;
			bf = (bf > max_b_sample) ? max_b_sample : bf;
			af = (af < min_a_sample) ? min_a_sample : af;
			af = (af > max_a_sample) ? max_a_sample : af;

			ri = (rf - (int)rf == 0) ? rf : (rf + 1);
			gi = (gf - (int)gf == 0) ? gf : (gf + 1);
			bi = (bf - (int)bf == 0) ? bf : (bf + 1);
			ai = (af - (int)af == 0) ? af : (af + 1);

			ri = (ri < 0) ? 0 : ri;
			ri = (ri > 255) ? 255 : ri;
			gi = (gi < 0) ? 0 : gi;
			gi = (gi > 255) ? 255 : gi;
			bi = (bi < 0) ? 0 : bi;
			bi = (bi > 255) ? 255 : bi;
			ai = (ai < 0) ? 0 : ai;
			ai = (ai > 255) ? 255 : ai;

			// (y, x+1)
			*(dstData + (3 * (bigW * y + (x + 1)))) = bi;
			*(dstData + (3 * (bigW * y + (x + 1)) + 1)) = gi;
			*(dstData + (3 * (bigW * y + (x + 1)) + 2)) = ri;

			for (int sx = -1; sx <= 2; ++sx)
			{
				for (int sy = -1; sy <= 2; ++sy)
				{
					// clamp pixel locations
					int csy = sx - sy + 1 + y;
					int csx = sx + sy - 1 + x;
					csy = (csy < 0) ? 0 : csy;
					csy = (csy > bigH - 1) ? bigH - 1 : csy;
					csx = (csx < 0) ? 0 : csx;
					csx = (csx > bigW - 1) ? bigW - 1 : csx;

					// sample & add weighted components
					// 3 is channel number
					uchar* bSample = dstData + (3 * (bigW * csy + csx));
					uchar* gSample = bSample + 1;
					uchar* rSample = bSample + 2;
					uchar aSample = 255;

					int offset = 4 * (sx + 1) + (sy + 1);
					*(rPtr + offset) = *rSample;
					*(gPtr + offset) = *gSample;
					*(bPtr + offset) = *bSample;
					*(aPtr + offset) = aSample;
					*(yPtr + offset) = rWeight[*rSample] + gWeight[*gSample] + bWeight[*bSample];
				}
			}
			d_edge = diagonal_edge(Y, wp);

			if (d_edge <= 0)
			{
				rf = w3 * (r[0][3] + r[3][0]) + w4 * (r[1][2] + r[2][1]);
				gf = w3 * (g[0][3] + g[3][0]) + w4 * (g[1][2] + g[2][1]);
				bf = w3 * (b[0][3] + b[3][0]) + w4 * (b[1][2] + b[2][1]);
				af = w3 * (a[0][3] + a[3][0]) + w4 * (a[1][2] + a[2][1]);
			}
			else
			{
				rf = w3 * (r[0][0] + r[3][3]) + w4 * (r[1][1] + r[2][2]);
				gf = w3 * (g[0][0] + g[3][3]) + w4 * (g[1][1] + g[2][2]);
				bf = w3 * (b[0][0] + b[3][3]) + w4 * (b[1][1] + b[2][2]);
				af = w3 * (a[0][0] + a[3][3]) + w4 * (a[1][1] + a[2][2]);
			}
			// anti-ringing, clamp.
			rf = (rf < min_r_sample) ? min_r_sample : rf;
			rf = (rf > max_r_sample) ? max_r_sample : rf;
			gf = (gf < min_g_sample) ? min_g_sample : gf;
			gf = (gf > max_g_sample) ? max_g_sample : gf;
			bf = (bf < min_b_sample) ? min_b_sample : bf;
			bf = (bf > max_b_sample) ? max_b_sample : bf;
			af = (af < min_a_sample) ? min_a_sample : af;
			af = (af > max_a_sample) ? max_a_sample : af;

			ri = (rf - (int)rf == 0) ? rf : (rf + 1);
			gi = (gf - (int)gf == 0) ? gf : (gf + 1);
			bi = (bf - (int)bf == 0) ? bf : (bf + 1);
			ai = (af - (int)af == 0) ? af : (af + 1);

			ri = (ri < 0) ? 0 : ri;
			ri = (ri > 255) ? 255 : ri;
			gi = (gi < 0) ? 0 : gi;
			gi = (gi > 255) ? 255 : gi;
			bi = (bi < 0) ? 0 : bi;
			bi = (bi > 255) ? 255 : bi;
			ai = (ai < 0) ? 0 : ai;
			ai = (ai > 255) ? 255 : ai;

			// (y+1, x)
			*(dstData + (3 * (bigW * (y + 1) + x))) = bi;
			*(dstData + (3 * (bigW * (y + 1) + x)) + 1) = gi;
			*(dstData + (3 * (bigW * (y + 1) + x)) + 2) = ri;
			++x;
		}
		++y;
	}
#pragma endregion

#pragma region Third Pass
	wp[0] = 2.0;
	wp[1] = 1.0;
	wp[2] = -1.0;
	wp[3] = 4.0;
	wp[4] = -1.0;
	wp[5] = 1.0;

	for (int y = bigH - 1; y >= 0; --y)
	{
		for (int x = bigW - 1; x >= 0; --x)
		{
			for (int sx = -2; sx <= 1; ++sx)
			{
				for (int sy = -2; sy <= 1; ++sy)
				{
					// clamp pixel locations
					int csy = sy + y;
					int csx = sx + x;
					csy = (csy < 0) ? 0 : csy;
					csy = (csy > bigH - 1) ? bigH - 1 : csy;
					csx = (csx < 0) ? 0 : csx;
					csx = (csx > bigW - 1) ? bigW - 1 : csx;

					// sample & add weighted components
					// 3 is channel number
					uchar* bSample = dstData + (3 * (bigW * csy + csx));
					uchar* gSample = bSample + 1;
					uchar* rSample = bSample + 2;
					uchar aSample = 255;

					int offset = 4 * (sx + 2) + (sy + 2);
					*(rPtr + offset) = *rSample;
					*(gPtr + offset) = *gSample;
					*(bPtr + offset) = *bSample;
					*(aPtr + offset) = aSample;
					*(yPtr + offset) = rWeight[*rSample] + gWeight[*gSample] + bWeight[*bSample];
				}
			}

			min_r_sample = getMin(r[1][1], r[2][1], r[1][2], r[2][2]);
			min_g_sample = getMin(g[1][1], g[2][1], g[1][2], g[2][2]);
			min_b_sample = getMin(b[1][1], b[2][1], b[1][2], b[2][2]);
			min_a_sample = getMin(a[1][1], a[2][1], a[1][2], a[2][2]);
			max_r_sample = getMax(r[1][1], r[2][1], r[1][2], r[2][2]);
			max_g_sample = getMax(g[1][1], g[2][1], g[1][2], g[2][2]);
			max_b_sample = getMax(b[1][1], b[2][1], b[1][2], b[2][2]);
			max_a_sample = getMax(a[1][1], a[2][1], a[1][2], a[2][2]);
			d_edge = diagonal_edge(Y, wp);

			if (d_edge <= 0)
			{
				rf = w1 * (r[0][3] + r[3][0]) + w2 * (r[1][2] + r[2][1]);
				gf = w1 * (g[0][3] + g[3][0]) + w2 * (g[1][2] + g[2][1]);
				bf = w1 * (b[0][3] + b[3][0]) + w2 * (b[1][2] + b[2][1]);
				af = w1 * (a[0][3] + a[3][0]) + w2 * (a[1][2] + a[2][1]);
			}
			else
			{
				rf = w1 * (r[0][0] + r[3][3]) + w2 * (r[1][1] + r[2][2]);
				gf = w1 * (g[0][0] + g[3][3]) + w2 * (g[1][1] + g[2][2]);
				bf = w1 * (b[0][0] + b[3][3]) + w2 * (b[1][1] + b[2][2]);
				af = w1 * (a[0][0] + a[3][3]) + w2 * (a[1][1] + a[2][2]);
			}

			// anti-ringing, clamp.
			rf = (rf < min_r_sample) ? min_r_sample : rf;
			rf = (rf > max_r_sample) ? max_r_sample : rf;
			gf = (gf < min_g_sample) ? min_g_sample : gf;
			gf = (gf > max_g_sample) ? max_g_sample : gf;
			bf = (bf < min_b_sample) ? min_b_sample : bf;
			bf = (bf > max_b_sample) ? max_b_sample : bf;
			af = (af < min_a_sample) ? min_a_sample : af;
			af = (af > max_a_sample) ? max_a_sample : af;

			ri = (rf - (int)rf == 0) ? rf : (rf + 1);
			gi = (gf - (int)gf == 0) ? gf : (gf + 1);
			bi = (bf - (int)bf == 0) ? bf : (bf + 1);
			ai = (af - (int)af == 0) ? af : (af + 1);

			ri = (ri < 0) ? 0 : ri;
			ri = (ri > 255) ? 255 : ri;
			gi = (gi < 0) ? 0 : gi;
			gi = (gi > 255) ? 255 : gi;
			bi = (bi < 0) ? 0 : bi;
			bi = (bi > 255) ? 255 : bi;
			ai = (ai < 0) ? 0 : ai;
			ai = (ai > 255) ? 255 : ai;
			
			// (y, x)
			*(dstData + (3 * (bigW * y + x))) = bi;
			*(dstData + (3 * (bigW * y + x)) + 1) = gi;
			*(dstData + (3 * (bigW * y + x)) + 2) = ri;
		}
	}
#pragma endregion

	return result;
}