#include "trackingDemo.h"

#include <opencv2\videoio.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\video.hpp>
#include <opencv2\core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

trackingDemo::trackingDemo()
{
}


trackingDemo::~trackingDemo()
{
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
	double, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
				color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

bool hasMvmt(double x, double y)
{
	return (abs(x - 0) > 2) || (abs(y-0) > 2);
}

bool isBlue(Vec3b col)
{
	return (col.val[0] > 150 && col.val[1] < 150 && col.val[2] < 150);
}

bool isYellow(Vec3b col)
{
	return (col.val[0] < 150 && col.val[1] > 150 && col.val[2] > 150);
}

static void drawKirby(const Mat& flow, Mat& cflowmap, Mat& bin, Mat& frame, int step, double, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f fxy = flow.at<Point2f>(y, x);
			Vec3b col = bin.at<Vec3b>(Point(x, y));
			col.val[0] = (hasMvmt(fxy.x, fxy.y)) ? 255 : 0;
			col.val[1] = (isBlue(frame.at<Vec3b>(Point(x, y)))) ? 255:0;
			col.val[2] = (isYellow(frame.at<Vec3b>(Point(x, y)))) ? 255:0;
			bin.at<Vec3b>(Point(x, y)) = col;
		}
}

int main(int argc, char** argv)
{
	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	Mat flow, cflow, frame;
	UMat gray, prevgray, uflow;
	namedWindow("flow", 1);

	for (;;)
	{
		cap >> frame;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		if (!prevgray.empty())
		{
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
			uflow.copyTo(flow);
			//drawOptFlowMap(flow, cflow, 4, 1.5, Scalar(0, 255, 0));
			Mat bin(cflow.rows, cflow.cols, CV_8UC3, Scalar(1,1,1));
			drawKirby(flow, cflow, bin, frame, 1, 1.5, CV_RGB(255, 0, 0));
			imshow("flow", bin);
		}
		if (waitKey(30) >= 0)
			break;
		std::swap(prevgray, gray);
	}
	return 0;
}