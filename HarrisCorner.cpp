//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <typeinfo>

using namespace std;
using namespace cv;

#define PI 3.1415926
#define WIN_SIZE 5

void ConvertMat( Mat src, Mat &dst );
void PrintMat( Mat &m );
void GrayScale( Mat image, Mat &gray );
void Convolution( Mat image, Mat &result, float ***filter, int window_size );
void GaussianFilter( float ***filter,  int window_size, float theta );
void Gradient( float ***filter_in, float ***filter_out, char var, int window_size );
void FindM( Mat &image, Mat &M, int x, int y, int window_size );

int width, height;

int main( int argc, char **argv)
{
	Mat image = imread( argv[1], 1 );
	Mat gray, gray2, gaus;
	Mat result_x, result_y;
	Mat draw;
	float **filter_gaus, **filter_x, **filter_y;

	GrayScale( image, gray );
	
	// ?????
	//cout << image.size[0] << " " << image.size[1] << " " << image.size[2]  << " " << gray.size[0] << " " << gray.size[1] << endl;
	
	/* opencv3 change: CV_RGB2GRAY => COLOR_RGB2GRAY */
	//cvtColor( image, gray, COLOR_BGR2GRAY );
	
	/* print filter
	for(int i=0;i<WIN_SIZE;i++){
		for(int j=0;j<WIN_SIZE;j++)
			cout << filter_gaus[i][j] << " ";
		cout << endl;
	}
	*/
	GaussianFilter( &filter_gaus, WIN_SIZE, 1 );
	
	Gradient( &filter_gaus, &filter_x, 'x', WIN_SIZE );
	Gradient( &filter_gaus, &filter_y, 'y', WIN_SIZE );
	
	Convolution( gray, gaus, &filter_gaus, WIN_SIZE );
	Convolution( gaus, result_x, &filter_x, WIN_SIZE );
	Convolution( gaus, result_y, &filter_y, WIN_SIZE );

	Mat Ixx = result_x.mul(result_x);
	Mat Iyy = result_y.mul(result_y);
	Mat Ixy = result_x.mul(result_y);

	Mat detM = Ixx.mul(Iyy) - Ixy.mul(Ixy);
	Mat trM = Ixx + Iyy;
	float k = 0.05;
	Mat R = detM - k * trM.mul(trM);
	/* print matrix */
	//PrintMat(gray);

	//cout << image.type() << endl;
	//cout << gray.type() << endl;
	//cout << gaus.type() << endl;
	//cout << result_x.type() << endl;

	free(filter_gaus);
	free(filter_x);
	free(filter_y);
	
	/* opencv gradient test */
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
	
	//PrintMat(gray);	
	/* cannot use opencv convertTo() successfully, so write another ConvertMat() */
	/*
	double minVal, maxVal;
	minMaxLoc(gray, &minVal, &maxVal);
	gray.convertTo(draw, CV_8U);
	*/
	ConvertMat( result_x, draw );
	PrintMat(result_x);	

	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	imshow( "Display Image", draw);

	waitKey(0);
	image.release();
	return 0;
}

//void ImageF2C()
//{
//}

void ConvertMat( Mat src, Mat &dst )
{
	dst.create( src.size(), CV_8U );
	for( int i=0 ; i < src.rows ; i++ )
	{
		for( int j=0 ; j < src.cols ; j++ )
		{
			dst.data[i*dst.step[0] + j*dst.step[1]] = (uchar)src.data[i*src.step[0] + j*src.step[1]];
		}
	}
}

void PrintMat( Mat &m )
{
	int row = m.rows;
	int col = m.cols;

	for( int i=0 ; i < row ; i++ )
	{
		for( int j=0 ; j < col ; j++ )
		{
			cout << (float)m.data[ i*m.step[0]+j*m.step[1] ] << " ";
		}
		cout << endl;
	}
}

/* compute the intensity of image */
void GrayScale( Mat image, Mat &gray )
{
	float sum;

	gray.create( image.rows, image.cols, CV_32FC1 );

	for( int i=0 ; i < image.rows ; i++ )
	{
		for( int j=0 ; j < image.cols ; j++ )
		{	
			sum  = 0.299 * (float)image.data[ i*image.step[0] + j*image.step[1] + 0*image.step[0] ];
			sum += 0.587 * (float)image.data[ i*image.step[0] + j*image.step[1] + 0*image.step[1] ];
			sum += 0.114 * (float)image.data[ i*image.step[0] + j*image.step[1] + 0*image.step[2] ];
			gray.data[ i*gray.step[0] + j*gray.step[1] ] = sum;
			//cout << sum << endl;
		}
	}
}

/* passing pixel on the edge will cause problem */
void FindM( Mat &image, Mat &M, int x, int y, int window_size )
{
	//float Ix = Gradient( image, x, y, 'x', window_size );
	//float Iy = Gradient( image, x, y, 'y', window_size );
	//cout << "[ " << Ix*Ix << " " << Ix*Iy << " ]" << endl;
	//cout << "[ " << Ix*Iy << " " << Iy*Iy << " ]" << endl;
	//M = ( Mat_<uchar>(2,2) << Ix*Ix, Ix*Iy, Ix*Iy, Iy*Iy );
}

/* do gradient on a filter */
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

/* set up Gaussian filter */
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

/* do convolution */
void Convolution( Mat image, Mat &result, float ***filter, int window_size )
{
	int col = image.cols;
	int row = image.rows;
	int half_ksize = window_size / 2;

	result.create( image.size(), image.type() );

	for( int i = 0 ; i < row ; i++ )
	{
		for( int j = 0 ; j < col ; j++ )
		{
			int up = -half_ksize, down = half_ksize;
			int left = -half_ksize, right = half_ksize;
			float sum = 0;
			float de = 0;

			if( i < half_ksize ) up = -i;
			if( i > row - half_ksize ) down = row - i - 1;
			if( j < half_ksize ) left = -j;
			if( j > col -half_ksize ) right = col -j -1;

			for( int y = up ; y <= down ; y++ )
			{
				for( int x = left ; x <= right ; x++ )
				{
					de += (*filter)[y+half_ksize][x+half_ksize];
					sum += (float)image.data[ (i+y)*image.step[0] + (j+x)*image.step[1] ] * (*filter)[y+half_ksize][x+half_ksize];
				}
			}
			result.data[ i*result.step[0] + j*result.step[1] ] = sum;// / de;
			//cout << sum / de << endl;
		}
	}
}
