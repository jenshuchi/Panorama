#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "feature.cpp"
#include "projection.cpp"

using namespace std;
using namespace cv;

#define PI 3.1415926

int main( int argc, char **argv )
{
	FILE *info_file;
	char file_path[50];
	char info_name[50];
	char line[50];
	char image_name[50];
	char tmp_str[100];
	int image_num;
	double *focal_length;
	vector<Mat> image;
	vector<Mat> intensity;
	vector< vector<Pix> > corner_list;
	vector< vector<Feature> > feature_list;

	strcpy( info_name, argv[2] );
	strcpy( file_path, argv[1] );
	strcpy( tmp_str, file_path );
	strcat( tmp_str, "/" );
	strcat( tmp_str, info_name );
	info_file = fopen( tmp_str, "r" );
	
	if( info_file == NULL )
		cout << "cannot find file" << endl;
	else
	{
		fgets( line, 50, info_file );
		sscanf( line, "%d", &image_num );
		focal_length = new double[image_num];
cout << "image_num = " << image_num << endl;

		for( int i=0 ; i < image_num ; i++ )
		{
			fgets( line, 50, info_file );
			sscanf( line, "%s %lf", image_name, &focal_length[i] );
			strcpy( tmp_str, file_path );
			strcat( tmp_str, "/" );
			strcat( tmp_str, image_name );
			
			image.push_back(Mat());
			intensity.push_back( Mat() );
			corner_list.push_back( vector<Pix>() );
			feature_list.push_back( vector<Feature>() );
			
			image[i] = imread( tmp_str, 1 );
			
			CornerDetection( image[i], intensity[i], corner_list[i] );
			CornerDescriptor( intensity[i], corner_list[i], feature_list[i], WIN_SIZE );

			cout << "XDDDD ~~ " << corner_list.size() << endl;
			Mat draw = image[i];
			
			vector<Pix> tmp_list = corner_list[i];
			cout << "XDDDD ~~ " << corner_list[i].size() << endl;
			for( int j=0 ; j < tmp_list.size() ; j++ )
			{
				circle( draw, tmp_list[j].first, 3, Scalar( 0, 0, 255 ), 1, 2 );
			}
			
			namedWindow( "Display Image", WINDOW_NORMAL );
			imshow( "Display Image", draw );
			//namedWindow( image_name, WINDOW_NORMAL );
			//imshow( image_name, draw );
			waitKey(0);
			
		}
	}

/*

	Mat draw = image;

	cout << corner_list.size() << endl;
	int *matching = TwoImageMatching( feature_list, feature_list);//, vector<Feature> &ftrs_2 )
	
	for( int i=0 ; i < sizeof(matching)/sizeof(int) ; i++ )
	{
		if(matching[i]!=-1)
			cout << feature_list[matching[i]].position << endl;
	}

	namedWindow( "Display Image", WINDOW_NORMAL );
	imshow( "Display Image", draw);

	waitKey(0);
*/

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
