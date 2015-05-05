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
#define WIN_SIZE 7

void HarrisCorner( Mat &image );
void GrayScale( Mat image, Mat &gray );
void Convolution( Mat image, Mat &result, double ***filter, int window_size );
void GaussianFilter( double ***filter,  int window_size, double theta );
void Gradient( double ***filter_in, double ***filter_out, char var, int window_size );
void FindCorners( Mat &R, vector<Point> &corner_list, double th, int window_size );
void PrintFilter( double ***filter, int window_size );
//void ConvertMat( Mat src, Mat &dst );
//void PrintMat( Mat &m );

void HarrisCorner( string file_name )
{	
	Mat image = imread( file_name, 1 );
	Mat rgb, gray, gray2, gaus;
	Mat result_x, result_y;
	Mat Ixx, Ixy, Iyy;
	Mat Sxx, Sxy, Syy;
	Mat detM, trM, R;
	Mat draw;
	double **filter_gaus, **filter_gaus2, **filter_x, **filter_y;
	double k = 0.04;
	vector<Point> corner_list;
	
	GrayScale( image, gray );
	//cvtColor( image, rgb, COLOR_BGR2GRAY );
	//rgb.convertTo(gray,CV_64FC1);


	GaussianFilter( &filter_gaus, WIN_SIZE, 0.84089642 );
	GaussianFilter( &filter_gaus2, WIN_SIZE, 3 );
	Gradient( &filter_gaus, &filter_x, 'x', WIN_SIZE );
	Gradient( &filter_gaus, &filter_y, 'y', WIN_SIZE );
	
	Convolution( gray, gaus, &filter_gaus, WIN_SIZE );
	Convolution( gaus, result_x, &filter_x, WIN_SIZE );
	Convolution( gaus, result_y, &filter_y, WIN_SIZE );

	Ixx = result_x.mul(result_x);
	Iyy = result_y.mul(result_y);
	Ixy = result_x.mul(result_y);
	
	Convolution( Ixx, Sxx, &filter_gaus2, WIN_SIZE );
	Convolution( Ixy, Sxy, &filter_gaus2, WIN_SIZE );
	Convolution( Iyy, Syy, &filter_gaus2, WIN_SIZE );

	detM = Sxx.mul(Syy) - Sxy.mul(Sxy);
	trM = Sxx + Syy;
	R = detM - k * trM.mul(trM);
/*
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(tmpM, &minVal, &maxVal, &minLoc, &maxLoc);
	//tmpM = tmpM - (double)minVal;//369.745;
	tmpM.convertTo( draw, CV_8U);//, 255.0/(maxVal-minVal),0);// -minVal*(maxVal-minVal) );
*/
	draw = image;
	FindCorners( R, corner_list, 500000000, WIN_SIZE );
	cout << corner_list.size() << " / " << draw.rows*draw.cols << endl;
	for( int i=0 ; i < corner_list.size() ; i++ )
	{
		circle( draw, corner_list[i], 1, Scalar( 0, 0, 255 ), 1, 2 );
	}

	namedWindow( "Display Image", WINDOW_NORMAL );
	imshow( "Display Image", draw);

	waitKey(0);
}
/* compute the intensity of image, and turn the type to double */
void GrayScale( Mat image, Mat &gray )
{
	double sum;
	
	gray.create( image.rows, image.cols, CV_64FC1 );

	for( int i=0 ; i < image.rows ; i++ )
	{
		for( int j=0 ; j < image.cols ; j++ )
		{	
			Vec3b rgb = image.at<Vec3b>(i,j);
			sum  = 0.299 * rgb[0];
			sum += 0.587 * rgb[1];
			sum += 0.114 * rgb[2];
			gray.at<double>(i,j) = sum;
		}
	}
}

/* do gradient on a filter */
void Gradient( double ***filter_in, double ***filter_out, char var, int window_size )
{
	int half_ksize = window_size / 2;
	double tmp, sum = 0;
	*filter_out = (double**)malloc( sizeof(double*) * window_size );

	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		(*filter_out)[i+half_ksize] = (double*)malloc( sizeof(double) * window_size );
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			if( var=='x' )
				(*filter_out)[i+half_ksize][j+half_ksize] = j * 1;//(*filter_in)[i+half_ksize][j+half_ksize];
			else if( var=='y' )
				(*filter_out)[i+half_ksize][j+half_ksize] = i * 1;//(*filter_in)[i+half_ksize][j+half_ksize];
		}
	}
	for( int i=0 ; i < window_size ; i++ )
	{
		for( int j=0 ; j < window_size ; j++ )
		{
			(*filter_out)[i][j] = (*filter_out)[i][j] / window_size ; 
		}
	}
}

/* set up Gaussian filter */
void GaussianFilter( double ***filter,  int window_size, double sigma )
{
	int half_ksize = window_size / 2;
	double tmp, sum = 0;
	*filter = (double**)malloc( sizeof(double*) * window_size ); /* *filter points to a 2D array */
	for( int i = -half_ksize ; i <= half_ksize ; i++ )
	{
		(*filter)[i+half_ksize] = (double*)malloc( sizeof(double) * window_size ); /* the parenthesis outer *filter are needed  */
		for( int j = -half_ksize ; j <= half_ksize ; j++ )
		{
			tmp = exp( -1*(i*i+j*j) / (2*sigma*sigma) ) / (2*PI*sigma*sigma);
			(*filter)[i+half_ksize][j+half_ksize] = tmp;
			sum += tmp;
		}
	}
	for( int i=0 ; i < window_size ; i++ )
		for( int j=0 ; j < window_size ; j++ )
			(*filter)[i][j] = (*filter)[i][j] / sum;
}

/* do convolution */
void Convolution( Mat image, Mat &result, double ***filter, int window_size )
{
	int row = image.rows;
	int col = image.cols;
	int half_ksize = window_size / 2;
	double max =0;

	result.create( image.size(), CV_64FC1 );

	for( int i = 0 ; i < row ; i++ )
	{
		for( int j = 0 ; j < col ; j++ )
		{
			result.at<double>(i,j) = image.at<double>(i,j);
			int up = -half_ksize, down = half_ksize;
			int left = -half_ksize, right = half_ksize;
			double sum = 0, de = 0;

			//if( i < half_ksize ) up = -i;
			//if( i > row - half_ksize ) down = row - i - 1;
			//if( j < half_ksize ) left = -j;
			//if( j > col - half_ksize ) right = col -j - 1;

			if( i >= half_ksize && i < row-half_ksize && j >= half_ksize & i < col-half_ksize )
			{

				for( int y = up ; y <= down ; y++ )
				{
					for( int x = left ; x <= right ; x++ )
					{
						de += (*filter)[y+half_ksize][x+half_ksize];
						sum += image.at<double>(i+y,j+x) * (*filter)[y+half_ksize][x+half_ksize];
					}
				}
				result.at<double>(i,j) = sum;
			}

		}
	}
}

void FindCorners( Mat &R, vector<Point> &corner_list, double th, int window_size )
{
	int col = R.cols;
	int row = R.rows;
	int half_ksize = window_size / 2;
	
	for( int i = half_ksize ; i < row - half_ksize ; )
	{
		for( int j = half_ksize ; j < col - half_ksize ; )
		{
			if( R.at<double>(i,j) >= th )
				corner_list.push_back(Point(j,i)); /* Point("j","i") for opencv "circle" to draw circle... */
			j += half_ksize;
		}
		i += half_ksize;
	}
}

void PrintFilter( double ***filter, int window_size )
{
	for( int i=0 ; i < window_size ; i++ )
	{
		for( int j=0 ; j < window_size ; j++ )
		{
			cout << (*filter)[i][j] << " ";
		}
		cout << endl;
	}
}
/*
void ConvertMat( Mat src, Mat &dst )
{
	double min_d, max_d;
	minMaxLoc( src, &min_d, &max_d );
	double min = (double)min_d;
	double range = (double)( max_d - min_d );

	dst.create( src.size(), CV_8UC1 );
	
	for( int i=0 ; i < src.rows ; i++ )
	{
		for( int j=0 ; j < src.cols ; j++ )
		{
			double tmp = src.at<double>(i,j);
			int tmp2 = (int)tmp;
			dst.at<uchar>(i,j) = (uchar)tmp2;
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
			cout << (int)m.data[ i*m.step[0]+j*m.step[1] ] << " ";
		}
		cout << endl;
	}
}
*/
