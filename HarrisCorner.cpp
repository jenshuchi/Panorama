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
void Convolution( Mat image, Mat &result, float ***filter, int window_size );
void GaussianFilter( float ***filter,  int window_size, float theta );
void Gradient( float ***filter_in, float ***filter_out, char var, int window_size );
void FindM( Mat &image, Mat &M, int x, int y, int window_size );

int width, height;

int main( int argc, char **argv)
{
	Mat image = imread( argv[1], 1 );
	Mat gray, gaus;
	Mat result_x, result_y;
	float **filter_gaus, **filter_x, **filter_y;
	
	// opencv3 change: CV_RGB2GRAY => COLOR_RGB2GRAY
	cvtColor( image, gray, COLOR_BGR2GRAY );
	
	GaussianFilter( &filter_gaus, 5, 1 );
	Gradient( &filter_gaus, &filter_x, 'x', 5 );
	Gradient( &filter_gaus, &filter_y, 'y', 5 );
	Convolution( gray, gaus, &filter_gaus, 5 );
	Convolution( gaus, result_x, &filter_x, 5 );
	Convolution( gaus, result_y, &filter_y, 5 );

	free(filter_gaus);
	free(filter_x);
	free(filter_y);
/*
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	
	Sobel( gaus, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel( gaus, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
*/


	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	imshow( "Display Image", gaus );

	waitKey(0);
	image.release();
	return 0;
}

int **HarrisCorner( Mat image )
{
}

// passing pixel on the edge will cause problem
void FindM( Mat &image, Mat &M, int x, int y, int window_size )
{
	//float Ix = Gradient( image, x, y, 'x', window_size );
	//float Iy = Gradient( image, x, y, 'y', window_size );
	//cout << "[ " << Ix*Ix << " " << Ix*Iy << " ]" << endl;
	//cout << "[ " << Ix*Iy << " " << Iy*Iy << " ]" << endl;
	//M = ( Mat_<uchar>(2,2) << Ix*Ix, Ix*Iy, Ix*Iy, Iy*Iy );
}

// do gradient on a filter
void Gradient( float ***filter_in, float ***filter_out, char var, int window_size )
{
	int half_ksize = window_size / 2;
	float tmp, sum = 0;
	*filter_out = (float**)malloc( sizeof(float*) * window_size );
	
	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		(*filter_out)[i+half_ksize] = (float*)malloc( sizeof(float) * window_size );
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			if( var=='x' )
				(*filter_out)[i+half_ksize][j+half_ksize] = (*filter_in)[i+half_ksize][j+half_ksize] * j;
			else if( var=='y' )
				(*filter_out)[i+half_ksize][j+half_ksize] = (*filter_in)[i+half_ksize][j+half_ksize] * i;
		}
	}
}

// set up Gaussian filter
void GaussianFilter( float ***filter,  int window_size, float theta )
{
	int half_ksize = window_size / 2;
	float tmp, sum = 0;
	*filter = (float**)malloc( sizeof(float*) * window_size ); /* *filter points to a 2D array */
	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		(*filter)[i+half_ksize] = (float*)malloc( sizeof(float) * window_size ); /* the parenthesis outer *filter are needed  */
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			tmp = exp( -1*(pow(i,2)+pow(j,2) ) / (2*pow(theta,2)) );
			(*filter)[i+half_ksize][j+half_ksize] = tmp;
			sum += tmp;
		}
	}
	for( int i=0 ; i < window_size ; i++ )
		for( int j=0 ; j < window_size ; j++ )
			(*filter)[i][j] = (*filter)[i][j] / sum;
}

void Convolution( Mat image, Mat &result, float ***filter, int window_size )
{
	int width = image.cols;
	int height = image.rows;
	int half_ksize = window_size / 2;
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
					sum += (float)image.data[ (i+y)*width + j+x ] * (*filter)[y+half_ksize][x+half_ksize];
				}
			}
//	cout << "gaus filter: ( " << i << ", " << j << " )" << endl;
			result.data[ i*width + j ] = sum;
		}
	}
}
