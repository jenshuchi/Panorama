//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#define PI 3.1415926

int **HarrisCorner( Mat image );
void GaussianFilter( Mat image, Mat &result, int window_size, float theta );
float Gradient( Mat &image, int x, int y, char var, int window_size );

int width, height;

int main( int argc, char **argv)
{
	Mat image = imread( argv[1], 1 );
	Mat gray, gua;
	// opencv3 change: CV_RGB2GRAY => COLOR_RGB2GRAY
	cvtColor( image, gray, COLOR_BGR2GRAY );
	GaussianFilter( gray, gua, 5, 3);

/*
	Mat C = (Mat_<uchar>(5,5) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
	for( int i=0 ; i < C.rows ; i++ )
	{
		for( int j=0 ; j < C.cols ; j++ )
		{
			cout << (int)C.data[i*C.cols+j] << " ";
		}
		cout << endl;
	}
	float diff = Gradient( C, 3, 3, 'x', 3 );	
	cout << diff << endl;
*/

	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	imshow( "Display Image", gua );

	waitKey(0);
	image.release();
	return 0;
}

int **HarrisCorner( Mat image )
{
}

// passing pixel on the edge will cause problem
float Gradient( Mat &image, int x, int y, char var, int window_size )
{
	int width = image.cols;
	int height = image.rows;
	int half_ksize = window_size / 2;
	float tmp, sum = 0;
	
	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			if( var=='x' )
				tmp = j;
			else if( var=='y' )
				tmp = i;
			sum += tmp * (float)image.data[ (y+i)*width + x+j ];
		}
	}
	return sum;
}

void GaussianFilter( Mat image, Mat &result, int window_size, float theta )
{
	int width = image.cols;
	int height = image.rows;
	int half_ksize = window_size / 2;

	// set up Gaussian filter
	float **filter = (float**)malloc( sizeof(float*) * window_size );
	float tmp, sum = 0;
	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		filter[i+half_ksize] = (float*)malloc( sizeof(float) * window_size );
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			tmp = exp( -1*(pow(i,2)+pow(j,2) ) / (2*pow(theta,2)) );
			filter[i+half_ksize][j+half_ksize] = tmp;
			sum += tmp;
		}
	}
	for( int i=0 ; i < window_size ; i++ )
		for( int j=0 ; j < window_size ; j++ )
			filter[i][j] = filter[i][j] / sum;

	result.create( image.size(), image.type() );

	// do convolution
	for( int i = half_ksize ; i < height - half_ksize ; i++ )
	{
		for( int j = half_ksize ; j < width - half_ksize ; j++ )
		{
			float sum = 0;
			for( int y = -half_ksize ; y <= half_ksize ; y++ )
			{
				for( int x = -half_ksize ; x <= half_ksize ; x++ )
				{
					sum += (float)image.data[ (i+y)*width + j+x ] * filter[y+half_ksize][x+half_ksize];
				}
			}
			result.data[ i*width + j ] = (uchar)sum;
		}
	}
	free(filter);
}
