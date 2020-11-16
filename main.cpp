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
		BORDER_CONSTANT, Scalar::all(0)); // ����
	
	Mat image = yantuoimage.clone();
	float* h=h1(M); // ��������
	/*for (int i = 0; i < M; i++)
	{
		cout << *(h + i) << ',';
	}*/
	Mat h_e = Mat::zeros(yantuoimage.size(), CV_32FC1);
	for (int i = 0; i < M; i++) // ��������һ��ѭ������ h_e 
	{
		float* he = h_e.ptr<float>(i);
		for (int j = 0; j < M; j++)
		{
			if (j <= i) { he[j] = *(h + i - j); }
			else { he[j] = *(h + M - j); }
		}
	}


	Mat n_e = Mat::zeros(yantuoimage.size(), CV_32FC1); // �������󣺳�ʼΪ0��û��������
	Mat tuihuaimage = Mat::zeros(yantuoimage.size(), CV_32FC1); // ����˻�����
	// cout << h_e.at<float>(3, 3) * 2 << endl;
	// cout << yantuoimage.at<float>(3, 3) << endl;
	// ��f����h����������ʹ�ü򵥵ľ����
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			float sum = 0;
			for (int k = 0; k < M; k++)
			{
				if (sum > 255) { sum = 255; }
				sum += h_e.at<float>(abs(i-j), abs(i-k)) * (float)yantuoimage.at<uchar>(j, k); // bug Խ��
			}
			// cout << sum << endl;
			tuihuaimage.at<float>(i, j) = sum + n_e.at<float>(i, j);
		}
	}

	//normalize(tuihuaimage, tuihuaimage, 0, 1, NORM_MINMAX);
	//imshow("tuihuaimage", tuihuaimage);

	Mat complexI1 = fourier(h_e, M, M); // ����ɢ����
	Mat complexI = fourier(tuihuaimage, M, M); // �˻�ͼ��

	//����Ƶ���˲��� ������˹��ͨ�˲��� ��ע����ʹ���˲���Ч�����ѣ�
	Mat butter_sharpen(tuihuaimage.size(), CV_32FC2); // 32λ������˫ͨ��
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

	multiply(complexI1, butter_sharpen, complexI1); // �������������ÿԪ�����ų˻���

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < M; j++)
		{
			if (complexI1.at<float>(i, j) == 0) { complexI1.at<float>(i, j) = 1e-3; }
			complexI1.at<float>(i, j) = complexI1.at<float>(i, j) * M * M;
		}
	}

	divide(complexI, complexI1, complexI); // ���� ������������� ������������������飿��


	idfft(complexI, image, M, M, "��ԭ");

	waitKey();
	return 0;
}

void zero_to_center(Mat& freq_plane)
{
	//    freq_plane = freq_plane(Rect(0, 0, freq_plane.cols & -2, freq_plane.rows & -2));
		//����Ϊʲô&��-2����鿴opencv�ĵ�
		//��ʵ��Ϊ�˰��к��б��ż�� -2�Ķ�������11111111.......10 ���һλ��0
	int cx = freq_plane.cols / 2; int cy = freq_plane.rows / 2;

	//���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����) �뺯��center_transform()������ͬ��ֻ��һ���ȴ���һ��dft���ٱ任
	Mat part1_r(freq_plane, Rect(0, 0, cx, cy));  //Ԫ�������ʾΪ(cx,cy)
	Mat part2_r(freq_plane, Rect(cx, 0, cx, cy));
	Mat part3_r(freq_plane, Rect(0, cy, cx, cy));
	Mat part4_r(freq_plane, Rect(cx, cy, cx, cy));

	Mat tmp;
	part1_r.copyTo(tmp);  //���������½���λ��(ʵ��)
	part4_r.copyTo(part1_r);
	tmp.copyTo(part4_r);

	part2_r.copyTo(tmp);  //���������½���λ��(ʵ��)
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
	//����Ҷ���任
	idft(complexI, complexI, DFT_INVERSE);

	Mat dstSharpen[2];
	split(complexI, dstSharpen); // ����ͨ�����л���Ϊ�����ͨ�����С�
	//    magnitude(dstSharpen[0],dstSharpen[1],dstSharpen[0]);  //���ֵ(ģ)
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

	threshold(dstSharpen[0], dstSharpen[0], 0, 255, THRESH_BINARY); // ��ֵ��
	normalize(dstSharpen[0], dstSharpen[0], 0, 1, NORM_MINMAX); // ӳ�䵽0~1
	dstSharpen[0] = dstSharpen[0](Rect(0, 0, image.cols, image.rows));
	imshow(str + s2, dstSharpen[0]);
}
Mat fourier(Mat padded, int oph, int opw)
{
	Mat temp[] = { padded, Mat::zeros(padded.size(),CV_32FC1) }; // ��ͨ��
	Mat complexI;
	merge(temp, 2, complexI); // �ϲ�����������γɵ�����ͨ������
	dft(complexI, complexI);    // ����Ҷ�任

	//��ʾƵ��ͼ
	split(complexI, temp);
	zero_to_center(temp[0]);
	zero_to_center(temp[1]);
	Mat aa;
	magnitude(temp[0], temp[1], aa); // ��������άʸ���ķ�ֵ��dst(I)=sqrt(x(I)^2+y(I)^2)
	divide(aa, oph * opw, aa); // ����
	imshow("pu", aa);

	merge(temp, 2, complexI);
	return complexI;
}