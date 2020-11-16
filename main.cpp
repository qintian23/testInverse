#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

#define mmax 10000

using namespace std;
using namespace cv;

float* h1(int n);
void zero_to_center(Mat& freq_plane);
Mat fourier(Mat padded, int oph, int opw);
void idfft(Mat complexI, Mat image, int oph, int opw, string str);

int main()
{
	const int cc = 1;
	char fname[] = "3.png";
	Mat imageIn = imread(fname, 0);
	if (imageIn.empty()) { cout << "Input failed!"; return 0; }
	int M = imageIn.cols, N = imageIn.rows;
	imshow("srcimage", imageIn);

	cout << M << ',' << N << endl;

	Mat yantuoimage;
	copyMakeBorder(imageIn, yantuoimage, 0, M - N, 0, 0,
		BORDER_CONSTANT, Scalar::all(0)); // 延拓
	
	Mat image = yantuoimage.clone();
	float* h=h1(M); // 线性算子
	/*for (int i = 0; i < M; i++)
	{
		cout << *(h + i) << ',';
	}*/
	Mat h_e = Mat::zeros(yantuoimage.size(), CV_32FC1);
	for (int i = 0; i < M; i++) // 首先生成一个循环矩阵 h_e 
	{
		float* he = h_e.ptr<float>(i);
		for (int j = 0; j < M; j++)
		{
			if (j <= i) { he[j] = *(h + i - j); }
			else { he[j] = *(h + M - j); }
		}
	}


	Mat n_e = Mat::zeros(yantuoimage.size(), CV_32FC1); // 噪声矩阵：初始为0（没有噪声）
	Mat tuihuaimage = Mat::zeros(yantuoimage.size(), CV_32FC1); // 输出退化矩阵
	// cout << h_e.at<float>(3, 3) * 2 << endl;
	// cout << yantuoimage.at<float>(3, 3) << endl;
	// 把f或者h，当做矩阵，使用简单的卷积。
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			float sum = 0;
			for (int k = 0; k < M; k++)
			{
				if (sum > 255) { sum = 255; }
				sum += h_e.at<float>(abs(i-j), abs(i-k)) * (float)yantuoimage.at<uchar>(j, k); // bug 越界
			}
			// cout << sum << endl;
			tuihuaimage.at<float>(i, j) = sum + n_e.at<float>(i, j);
		}
	}

	//normalize(tuihuaimage, tuihuaimage, 0, 1, NORM_MINMAX);
	//imshow("tuihuaimage", tuihuaimage);

	Mat complexI1 = fourier(h_e, M, M); // 点扩散函数
	Mat complexI = fourier(tuihuaimage, M, M); // 退化图像

	//生成频域滤波核 巴特沃斯低通滤波器 （注：不使用滤波器效果更佳）
	Mat butter_sharpen(tuihuaimage.size(), CV_32FC2); // 32位浮点型双通道
	double D0 = 50.0;
	int n = 4;
	for (int i = 0; i < butter_sharpen.rows; i++) {
		float* q = butter_sharpen.ptr<float>(i);
		for (int j = 0; j < butter_sharpen.cols; j++) 
		{
			double d = sqrt(pow(i - M / 2, 2) + pow(j - M / 2, 2));
			q[2 * j] = 1.0 / (1 + pow(D0 / d, 2 * n));
			q[2 * j + 1] = 1.0 / (1 + pow(D0 / d, 2 * n));
		}
	}

	multiply(complexI1, butter_sharpen, complexI1); // 计算两个数组的每元素缩放乘积。

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (complexI1.at<float>(i, j) == 0) { complexI1.at<float>(i, j) = 1e-3; }
			complexI1.at<float>(i, j) = complexI1.at<float>(i, j) * M * M;
		}
	}

	divide(complexI, complexI1, complexI); // 除法 会有无穷大的情况 ，怎样解决这样的事情？？


	idfft(complexI, image, M, M, "复原");

	waitKey();
	return 0;
}

void zero_to_center(Mat& freq_plane)
{
	//    freq_plane = freq_plane(Rect(0, 0, freq_plane.cols & -2, freq_plane.rows & -2));
		//这里为什么&上-2具体查看opencv文档
		//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	int cx = freq_plane.cols / 2; int cy = freq_plane.rows / 2;

	//以下的操作是移动图像  (零频移到中心) 与函数center_transform()作用相同，只是一个先处理，一个dft后再变换
	Mat part1_r(freq_plane, Rect(0, 0, cx, cy));  //元素坐标表示为(cx,cy)
	Mat part2_r(freq_plane, Rect(cx, 0, cx, cy));
	Mat part3_r(freq_plane, Rect(0, cy, cx, cy));
	Mat part4_r(freq_plane, Rect(cx, cy, cx, cy));

	Mat tmp;
	part1_r.copyTo(tmp);  //左上与右下交换位置(实部)
	part4_r.copyTo(part1_r);
	tmp.copyTo(part4_r);

	part2_r.copyTo(tmp);  //右上与左下交换位置(实部)
	part3_r.copyTo(part2_r);
	tmp.copyTo(part3_r);
}
float* h1(int n)
{
	float* hh = NULL;
	float harray[mmax];
	hh = harray;
	for (int i = 0; i < n; i++)
	{
		hh[i] = (i & 1) > 0 ? 1 : -1;
	}
	return hh;
}
void idfft(Mat complexI, Mat image, int oph, int opw, string str)
{
	string s1 = "_filter", s2 = "dstSharpen";
	//傅里叶反变换
	idft(complexI, complexI, DFT_INVERSE);

	Mat dstSharpen[2];
	split(complexI, dstSharpen); // 将多通道阵列划分为多个单通道阵列。
	//    magnitude(dstSharpen[0],dstSharpen[1],dstSharpen[0]);  //求幅值(模)
	for (int i = 0; i < oph; i++) {
		float* q = dstSharpen[0].ptr<float>(i);
		for (int j = 0; j < opw; j++) {
			q[j] = q[j] * pow(-1, i + j);
		}
	}
	Mat show;
	//    divide(dstSharpen[0], dstSharpen[0].cols*dstSharpen[0].rows, show);
	normalize(dstSharpen[0], show, 0, 1, NORM_MINMAX); //....
	show = show(Rect(0, 0, image.cols, image.rows));
	imshow(str + s1, show);

	threshold(dstSharpen[0], dstSharpen[0], 0, 255, THRESH_BINARY); // 阈值化
	normalize(dstSharpen[0], dstSharpen[0], 0, 1, NORM_MINMAX); // 映射到0~1
	dstSharpen[0] = dstSharpen[0](Rect(0, 0, image.cols, image.rows));
	imshow(str + s2, dstSharpen[0]);
}
Mat fourier(Mat padded, int oph, int opw)
{
	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) }; // 二通道
	Mat complexI;
	merge(temp, 2, complexI); // 合并多个阵列以形成单个多通道阵列
	dft(complexI, complexI);    // 傅里叶变换

	//显示频谱图
	split(complexI, temp);
	zero_to_center(temp[0]);
	zero_to_center(temp[1]);
	Mat aa;
	magnitude(temp[0], temp[1], aa); // 计算计算二维矢量的幅值。dst(I)=sqrt(x(I)^2+y(I)^2)
	divide(aa, oph * opw, aa); // 除法
	imshow("pu", aa);

	merge(temp, 2, complexI);
	return complexI;
}