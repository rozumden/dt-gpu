#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <vector>
#include "include.hpp"
using namespace std;
using namespace cv;
#define REPEAT 10

void visualise(cv::Mat &dt, cv::Mat &lm) {
	Mat ones(dt.size(), dt.type());
    ones.setTo(1);

    Mat lmd;
	lm.convertTo(lmd, dt.type());

    double min, max;
	cv::minMaxLoc(dt, &min, &max);
    Mat normdt = dt / max;
 	
	std::vector<cv::Mat> channels;
    Mat mask = 1 - (lmd/255);
    Mat out = mask.mul(normdt);
    Mat out2 = (ones+lmd).mul(normdt);
    channels.push_back(out);
    channels.push_back(out);
    channels.push_back(out2);
    Mat vis;
    cv::merge(channels, vis);

    namedWindow( "Display window", WINDOW_NORMAL ); // Create a window for display.
    imshow( "Display window", vis);    

    waitKey(0); // Wait for a keystroke in the window
}

void process(string name, int distmask, bool gpu) {
	Mat image = imread(name, IMREAD_GRAYSCALE);
    Size sz = image.size();
    cout << "Processing image " << name << endl;
    cout << "width: " << sz.width << ", height: " << sz.height << endl;
    Mat bin(sz, DataType<unsigned char>::type);
    threshold(image, bin, 128, 255, cv::THRESH_BINARY);
    Mat dt(sz, CV_32F);
    Mat lm(sz, DataType<unsigned char>::type);
    if(gpu) {
    	cout << "On GPU..." << endl;
    	dt_lm_gpu(bin, dt, lm, distmask);
    } else {
    	cout << "On CPU..." << endl;
    	distance_transform(bin, dt, distmask);
	    local_maxima(dt, lm);
    }
    visualise(dt, lm);
}

void test(string name, int distmask, bool vis) {
	Mat image = imread(name, IMREAD_GRAYSCALE);
    Size sz = image.size();
    cout << "Testing image " << name << endl;
    cout << "width: " << sz.width << ", height: " << sz.height << endl;

    Mat bin(sz, DataType<unsigned char>::type);
    threshold(image, bin, 128, 255, cv::THRESH_BINARY);

    std::chrono::duration<double> elapsed;
    double accCpu = 0;
    double accGpu = 0;
    for(int e = 0; e < REPEAT; e++) {
	    // CPU
	    Mat dt(sz, CV_32F);
	    Mat lm(sz, DataType<unsigned char>::type);
	    auto start = std::chrono::high_resolution_clock::now();
	    distance_transform(bin, dt, distmask);
	    local_maxima(dt, lm);
		auto finish = std::chrono::high_resolution_clock::now();
		elapsed = finish - start;
		if(vis) std::cout << "[CPU] Elapsed time: " << elapsed.count() << " s\n";
		accCpu += elapsed.count();

		// GPU
	    Mat dt_gpu(sz, CV_32F);
	    Mat lm_gpu(sz, DataType<unsigned char>::type);
	    start = std::chrono::high_resolution_clock::now();
	    dt_lm_gpu(bin, dt_gpu, lm_gpu, distmask);
	    finish = std::chrono::high_resolution_clock::now();
		elapsed = finish - start;
		if(vis) std::cout << "[GPU] Elapsed time: " << elapsed.count() << " s\n";
		accGpu += elapsed.count();

		Mat diff = cv::abs(dt - dt_gpu);
		int diffdt = cv::countNonZero(diff > 0.001);
		double min, maxdiffdt;
		cv::minMaxLoc(diff, &min, &maxdiffdt);
		int difflm = cv::countNonZero(lm != lm_gpu);
		if(vis) cout << "Number of wrong DT pixels: " << diffdt << ", max difference: " << maxdiffdt << endl;
		if(vis) cout << "Number of wrong LM pixels: " << difflm << endl;
		if(vis) cout << "-------------------------------------------" << endl;
		if(vis) {
    		visualise(dt_gpu, lm_gpu);
    		return;
    	}
	}
	cout << "[CPU] MEAN TIME OF " << REPEAT << " REPEATS is " <<  accCpu / REPEAT << endl;
	cout << "[GPU] MEAN TIME OF " << REPEAT << " REPEATS is " <<  accGpu / REPEAT << endl;
	cout << "-------------------------------------------" << endl;
}

int main(int argc, char const *argv[])
{
	if(argc > 1) {
		int distmask = DIST_MASK_PRECISE;
		int gpu = 1;
		if(argc > 2) 
			distmask = atoi(argv[2]);
		if(argc > 3)
			gpu = atoi(argv[3]);
		
		process(argv[1], distmask, gpu);
		return 0;
	} 

	vector<string> str;
	str.push_back("man.jpg");
	str.push_back("io.png");
	str.push_back("stripes.jpg");
	str.push_back("pattern.jpg");
	
	// for testing
	test(str[0], DIST_MASK_3, false);

	cout << "Using 3x3 approximation mask" << endl;
	for(int i = 0; i < str.size(); i++) {
		cout << "-------------------------------------------" << endl;
		string s = str[i];
		test(s, DIST_MASK_3, false);
		cout << "-------------------------------------------" << endl;
	}

	cout << "Using 5x5 approximation mask" << endl;
	for(int i = 0; i < str.size(); i++) {
		cout << "-------------------------------------------" << endl;
		string s = str[i];
		test(s, DIST_MASK_5, false);
		cout << "-------------------------------------------" << endl;
	}

	cout << "Using precise distance tranform" << endl;
	for(int i = 0; i < str.size(); i++) {
		cout << "-------------------------------------------" << endl;
		string s = str[i];
		test(s, DIST_MASK_PRECISE, false);
		cout << "-------------------------------------------" << endl;
	}

	return 0;
}