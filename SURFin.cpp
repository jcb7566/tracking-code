/*
 * SURFin.cpp
 *	Implementation of FeatureTracking class on a series of images.
 *
 *  Created on: Feb 15, 2013
 *      Author: jcb7566
 */

#include "FeatureTracking.h"

using namespace std;
using namespace cv;

int main(){
	//Mat outImage;

	FeatureTracking tracker;
	if(!tracker.havePastImage()){
	Mat image1 = imread("AntCenter.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if(!image1.data){
		cout << "---------->Error: No Image Found" << endl;
		return 0;
	}
	tracker.initTrack(image1,4000);
	}

	Mat nextImage = imread("AntCenter2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if(!nextImage.data){
		cout << "---------->Error: No Image Found" << endl;
		return 0;
	}
	if(tracker.havePastImage()){
	tracker.trackFeatures(nextImage);
	}

	return 1;

}


