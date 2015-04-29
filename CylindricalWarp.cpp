#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.1415926

void readImgs( vector<string> filenames, vector<Mat>& dst, vector<int>& fls ); /* vector<Mat>"&" means reference */
void cylinWarp( Mat &dst, Mat src, int f );

void readImgs( vector<string> filenames, vector<Mat>& dst, vector<int>& fls )
{
	for( int i=0 ; i < filenames.size() ; i++)
	{
		Mat img = imread( filenames[i], 1 );
		fls[i] = img.cols * 4.4 / 6.16 ;
		dst.push_back(img);
	}
}

void cylinWarp( Mat &dst, Mat src, int f )
{
	int width = src.cols;
	int height = src.rows;
	float Ox = width / 2;
	float Oy = height / 2;
	float x, y, x_p, y_p;
	float s = f;
	float theta;
	float h;
	
	src.copyTo(dst);
	
	/* initialize */
	Vec3b z;
	z[0] = (uchar)0; z[1] = (uchar)0; z[2] = (uchar)0;
	for( int x=0 ; x < width ; x++ )
		for( int y=0 ; y < height ; y++ )
			dst.at<Vec3b>(Point(x,y)) = z;

	for( int i=0 ; i < width ; i++ )
	{
		for( int j=0 ; j < height ; j++ )
		{
			x = i - Ox;
			y = j - Oy;
			theta = atan(x/f);// * 180 / PI;
			h = y / sqrt( x*x + f*f );
			x_p = s * theta;
			y_p = s * h ;
			int x_d = (int)(x_p+Ox);
			int y_d = (int)(y_p+Oy);
			if( x_d >= 0 && y_d >= 0 && x_d < width && y_d < height )
				dst.at<Vec3b>( y_d, x_d) = src.at<Vec3b>(j,i);
		}
	}
}
