#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

// Apply k-means algorithm to find 'object_count' objects in the image.
vector<Mat> run_k_means(const Mat& img, int object_count) {
	// Create a data, where for each pixel we have 5 columns, their coordinates and RGB colors.
	Mat data(img.rows * img.cols, 5, CV_32F);

	// These 2 values can be fine tuned to make the algorithm work better.
	double max_color_distance = 255;
	double max_spatial_distance = std::sqrt(img.rows * img.rows + img.cols * img.cols);

	int k = 0;
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3b intensity = img.at<Vec3b>(i, j);
			data.at<float>(k, 0) = intensity[0] / max_color_distance;
			data.at<float>(k, 1) = intensity[1] / max_color_distance;
			data.at<float>(k, 2) = intensity[2] / max_color_distance;
			data.at<float>(k, 3) = i / max_spatial_distance;
			data.at<float>(k, 4) = j / max_spatial_distance;
			++k;
		}
	Mat labels, centers;
	cv::kmeans(
		data, // data which we want to cluster.
		object_count, // number of clusters to split by.
		labels, // shows cluster indices for each pixel.
		TermCriteria(CV_TERMCRIT_ITER, 100, 1.0),  // terminate after 100 iterations or after the loss function value < 1.0
		3, // number of attempts of the algorithm with different starting points of centroids.
		KMEANS_PP_CENTERS, 
		centers // centroids of clusters created.
	);

	// Now let's create a separate image for each object/region found.
	vector<Mat> results;
	
	for (int o = 0; o < object_count; ++o) {
		Mat result = img.clone();
		int k = 0;
		for (int i = 0; i < img.rows; ++i)
			for (int j = 0; j < img.cols; ++j)
			{
				if (labels.at<int>(k) != o) {
					// Make it black.
					result.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
				++k;
			}
		results.push_back(std::move(result));
	}
	return results;
}

// A function to display multiple images in the same window,
// copied from the internet.
void display_multiple_images_in_one_window(string title, int nArgs, ...) {
	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row
	// h - Maximum number of images in a column
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying
	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 14) {
		printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
		return;
	}
	// Determine the size of the image,
	// and the number of rows/cols
	// from number of arguments
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		Mat img = va_arg(args, Mat);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img.empty()) {
			printf("Invalid arguments");
			return;
		}

		// Find the width and height of the image
		x = img.cols;
		y = img.rows;

		// Find whether height or width is greater in order to resize the image
		max = (x > y) ? x : y;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		// Set the image ROI to display the current image
		// Resize the input image and copy the it to the Single Big Image
		Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
		Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	// Create a new window, and show the Single Big Image
	namedWindow(title, 1);
	imwrite("ResultingImage.jpg", DispImage);
	imshow(title, DispImage);
	waitKey();

	// End the number of arguments
	va_end(args);
}
int main()
{
	Mat img1 = imread("Fish.png");
	
	int partition_count = 4;
	// Divide image into 3 parts. I hope that one of the 3 will be my fish.
	std::vector<Mat> results = run_k_means(img1, partition_count);

	display_multiple_images_in_one_window(
		"Steps of Basic region growing", partition_count + 1,
		img1, results[0], results[1], results[2], results[3]);

	waitKey(0);
	return 0;
}