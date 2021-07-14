#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>
#include <stdio.h>


using namespace cv;
using namespace std;
using namespace std::chrono;

int P_score = 25;						//Parameter for check black welding.
int P_canny_forward = 110;				//Parameter for value forward in canny.
int P_canny_backward = 204;				//Parameter for value backward in canny.

void show_histogram(string const& name, Mat1b const& image)
{
	float max = 0;
	int sum_col = 0;
	int sum_AVG = 0;

	for (int i = 0; i < image.cols; i++)
	{
		int column_sum = 0;
		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);

			if (column_sum > max) {
				max = column_sum;
			}
		}
		sum_col += column_sum;
	}
	sum_AVG = sum_col / (image.cols * image.rows);
	
	// Set histogram bins count
	int bins = image.cols;
	// Set ranges for histogram bins
	float lranges[] = { 0, bins };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	Mat hist;
	int channels[] = { 0 };
	float maxN = max / 50;
	float const hist_height = maxN;
	Mat3b hist_image = Mat3b::zeros(hist_height + 10, bins + 20);

	for (int i = 0; i < image.cols; i++)
	{
		float column_sum = 0;
		for (int k = 0; k < image.rows; k++)
		{
			column_sum += image.at<unsigned char>(k, i);
		}
		float const height = cvRound(column_sum * hist_height / max);
		line(hist_image, Point(i + 10, (hist_height - height) + 50), Point(i + 10, hist_height), Scalar::all(255));
	}
	cout << " Average : " << sum_AVG << endl;

	Mat canny_output;
	Canny(hist_image, canny_output, 50, 50 * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// create hull array for convex hull points
	vector< vector<Point> > hull(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], 1);
	}
	drawContours(hist_image, contours, 0, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
	if ( sum_AVG > P_score) {
		cout << " Defect Detection  " << endl;
		cout << "===================" << endl;
	}
	else {
		cout << " Non defect Detection  " << endl;
		cout << "===================" << endl;
	}
	imshow(name, hist_image);
	//This Code tell runtime this program.
	vector<int> values(10000);
	auto f = []() -> int { return rand() % 10000; };
	generate(values.begin(), values.end(), f);
	auto start = high_resolution_clock::now();
	sort(values.begin(), values.end());
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	cout << "Time taken by function: " << duration.count() << " milliseconds" << endl;
}


int Black_Welding(Mat imageOriginal) {
	Mat imgG, imgC, imgRz, imgTh, imgSobely;;
	int status = 0;
	imgRz = imageOriginal.clone();			//Copy image to imgRz.
	Mat ddst, cdst,cdstP;
	cvtColor(imgRz, imgG, COLOR_BGR2GRAY);
	Canny(imgRz, ddst, P_canny_forward, P_canny_backward, 3);
	cvtColor(ddst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();
	imshow("Gray", cdst);
	vector<Vec4i> linesP;
	HoughLinesP(ddst, linesP, 1, CV_PI / 180, 120, 10, 300); // runs the actual detection
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(imgRz, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, 4);
	}
	Vec4i L = linesP[0];
	Point p1 = Point(L[0], L[1]), p2 = Point(L[2], L[3]);
	// find RotatedRect for lines
	vector<Point2f> pts1;
	pts1.push_back(p1);
	pts1.push_back(p2);

	RotatedRect rr1 = minAreaRect(pts1);
	// change width or height of the RotatedRect
	if (rr1.size.width)
		rr1.size.height = 80;
	else
		rr1.size.width = 80;
	// Draw the RotatedRect
	Point2f vtx[4];
	Point2f vtxC[4];

	rr1.points(vtx);
	for (int i = 0; i < 4; i++) {
		line(imgRz, vtx[i], vtx[(i + 1) % 4], Scalar(0, 0, 255), 1);
	}
	imshow("RE", imgRz);
	//Warp image. 
	Mat imgW, matrix;
	float w = 630, h = 80;
	Point2f src[4] = { vtx[3],vtx[0],vtx[2],vtx[1] };
	Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };
	matrix = getPerspectiveTransform(src, dst);
	warpPerspective(imgG, imgW, matrix, Point(w, h));
	//cout << "matrix " << Point(w, h) << endl;
	imshow("W", imgW);
	Rect crop_region(40, 0, 540, 80);
	imgC = imgW(crop_region);
	Rect myROI(10, 20, 200, 50);
	Mat croppedRef(imgC, myROI);
	Mat imgCrop;
	// Copy the data into new matrix
	croppedRef.copyTo(imgCrop);
	imshow("REz", imgCrop);

	resize(imgCrop, imgCrop, Size(imgCrop.cols / 2, imgCrop.rows / 2));
	Sobel(imgCrop, imgSobely, CV_8U, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	rotate(imgSobely, imgSobely, ROTATE_90_COUNTERCLOCKWISE);
	imshow("Sobel convert ", imgSobely);
	show_histogram("Histogram image", imgSobely);
	return status;
}


int main(int argc, const char* argv[]) {
	Mat imgOri, imgRz;
	int result = 0;
	string folder("img/*.jpg");
	vector<String> fn;
	glob(folder, fn, false);
	vector<Mat> images;
	size_t count = fn.size(); //number of png files in images folder.\

	//Check number of images.
	cout << "image in folder  " << count << endl;

	 P_score = 25;							//Parameter for check black welding.
	 P_canny_forward = 110;					//Parameter for value forward in canny.
	 P_canny_backward = 204;				//Parameter for value backward in canny.

	//Main LooB.
	for (size_t i = 0; i < count; i++)
	{
		//Preprocessing
		imgOri = imread(fn[i]);
		resize(imgOri, imgRz, Size(), 0.5, 0.5); //Half Resize 1280*1040 to 640*520 pixcel.

		result = Black_Welding(imgRz);			//Call function check black welding problem.
		waitKey(0);
	}
	waitKey(0);
	return 0;
}
