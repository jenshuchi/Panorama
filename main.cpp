#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "feature.cpp"
#include "projection.cpp"

using namespace std;
using namespace cv;

#define PI 3.1415926

int main( int argc, char **argv )
{
	/* for harris corner */
	
	   Mat image = imread( argv[1], 1 );
	   vector<Point> feature_list;
	   CornerDetection( image, feature_list );

	   Mat draw = image;

	   for( int i=0 ; i < feature_list.size() ; i++ )
	   {
	   circle( draw, feature_list[i], 1, Scalar( 0, 0, 255 ), 1, 2 );
	   }

	   namedWindow( "Display Image", WINDOW_NORMAL );
	   imshow( "Display Image", draw);

	   waitKey(0);
	 

	/* for cylindrical warping */
	/*
	int img_num = 2;
	vector<Mat> imgs;
	vector<string> filenames( img_num );
	vector<int> fl( img_num );

	filenames[0] = "house1.jpg";
	filenames[1] = "house2.jpg";

	readImgs( filenames, imgs, fl );
	cout << fl[0] << endl;

	Mat cy_img;
	cylinWarp( cy_img, imgs[0], fl[0] );

	//for(int i=0 ; i < cy_img.cols ; i++ )
	//	for(int j=0 ; j < cy_img.rows ; j++ )
	//		cout << cy_img.at<Vec3b>(j,i) << endl;
	cout << cy_img.at<Vec3b>(100,100) << endl;

	namedWindow( "Display Image", WINDOW_NORMAL );
	imshow( "Display Image", cy_img );
	namedWindow( "Display Image2", WINDOW_NORMAL );
	imshow( "Display Image2", imgs[0] );
	waitKey(0);
	*/

	/* for test... */
	/*
	   int mat_size = 100;
	   rgb.create(mat_size,mat_size,CV_64FC3);
	   gray.create(mat_size,mat_size,CV_64FC1);
	   for(int i=0;i<mat_size;i++)
	   {
	   for(int j=0;j<mat_size;j++)
	   {
	   Vec3d pix;
	   pix[0] = 0.0; pix[1] = 0.0; pix[2] = 0.0;
	   rgb.at<Vec3d>(i,j)= pix;
	   gray.at<double>(i,j)= 0.0;
	   }
	   }
	for(int i=10;i<59;i++)
	{
		for(int j=80;j<89;j++)
		{
			Vec3d pix;
			pix[0] = 255.0; pix[1] = 255.0; pix[2] = 255.0;
			rgb.at<Vec3d>(i,j)= pix;
			gray.at<double>(i,j)= 255.0;
		}
	}
	*/
	 
	return 0;
}

/*
/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char source_window[30] = "Source image";
char corners_window[30] = "Corners detected";

/// Function header
void cornerHarris_demo( int, void* );

//  @function main
int main( int argc, char** argv )
{
	/// Load source image and convert it to gray
	src = imread( argv[1], 1 );
	cvtColor( src, src_gray, COLOR_BGR2GRAY );

	/// Create a window and a trackbar
	namedWindow( source_window, WINDOW_AUTOSIZE );
	createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
	imshow( source_window, src );

	cornerHarris_demo( 0, 0 );

	waitKey(0);
	return(0);
}

//   @function cornerHarris_demo
void cornerHarris_demo( int, void* )
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros( src.size(), CV_32FC1 );

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizing
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	/// Drawing a circle around corners
	for( int j = 0; j < dst_norm.rows ; j++ )
	{ for( int i = 0; i < dst_norm.cols; i++ )
		{
			if( (int) dst_norm.at<float>(j,i) > thresh )
			{
				circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
			}
		}
	}
	/// Showing the result
	namedWindow( corners_window, WINDOW_AUTOSIZE );
	imshow( corners_window, dst_norm_scaled );
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
