/*
 * FeatureTracking.h
 *
 *  Created on: Feb 15, 2013
 *      Author: jcb7566
 */

#ifndef FEATURETRACKING_H_
#define FEATURETRACKING_H_

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "cmath"

class FeatureTracking{
public:
	bool havePastImage();
	void initTrack(cv::Mat &, double);
	void trackFeatures(cv::Mat &);

private:
	double minDist;
	double maxDist;

	bool haveTrack;
	bool pastImageExists;

	cv::Mat initDescriptors;
	cv::Mat newDescriptors;
	cv::Mat trackDescriptors;
	cv::Mat oldImage;

	std::vector<cv::KeyPoint> initKeypoints;
	std::vector<cv::KeyPoint> newKeypoints;
	std::vector<cv::KeyPoint> trackKeypoints;

	cv::SurfFeatureDetector surf;
	cv::SurfDescriptorExtractor extractor;
	cv::FlannBasedMatcher matcher;


};



#endif /* FEATURETRACKING_H_ */
