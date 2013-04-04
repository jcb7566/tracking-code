/*
 * FeatureTracking.cpp
 *
 *	This class is used to detect stable features and descriptors in an image according to the SURF algorithm.
 *	The stable features/descriptors are carried over to the next frame. A matching algorithm is run to find matches of
 *	descriptors in the current frame to descriptors in the last frame. Matched descriptors/features are passed out to
 *	be tracked in subsequent frames.
 *
 *  Created on: Feb 14, 2013
 *      Author: jcb7566
 */

#include "FeatureTracking.h"

using namespace std;
using namespace cv;

bool FeatureTracking::havePastImage(){

	return pastImageExists;
}

void FeatureTracking::initTrack(Mat &image, double thresh){
	surf.hessianThreshold = thresh;						//Set SURF threshold

	surf.detect(image,initKeypoints);
	extractor.compute(image,initKeypoints,initDescriptors);
	oldImage = image;
	pastImageExists = true;
	drawKeypoints(image, initKeypoints,image,Scalar(255,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	namedWindow("SURF Features");
	imshow("SURF Features",image);
	waitKey(0);
}

void FeatureTracking::trackFeatures(Mat &image){
	/*SurfFeatureDetector surf(2500);								//Create surf object
	SurfDescriptorExtractor extractor;							//descriptor extractor object
	FlannBasedMatcher matcher;	*/								//creates matcher object

	double minDist = 100;
	double maxDist = 0;
	vector<DMatch>	matches;
	vector<DMatch> goodMatches;
	vector<KeyPoint> trackold;

	if(!haveTrack && pastImageExists){							//if I don't have matched features to track, do:

		surf.detect(image,newKeypoints);						//Detect key points with SURF
		extractor.compute(image,newKeypoints,newDescriptors);	//Calculate Descriptors

		//cerr << "New keypoints: " << newKeypoints.size() << endl;
		//cerr << "New descriptors: " << newDescriptors.rows << endl;

		matcher.match(newDescriptors,initDescriptors,matches);	//Match descriptors

		//cerr << "Matches: " << matches.size() << endl;
		/*for(int i=0;i< (int)matches.size();i++){
			cerr << "Match: " << matches[i].queryIdx << "--" << matches[i].trainIdx << endl;
		}*/

		//Quick calculation of max and min distances between key points
		for( int i = 0; i < (int)matches.size(); i++ )
		  { double dist = matches[i].distance;					//load element of matches vector
			if( dist < minDist ) minDist = dist;
			if( dist > maxDist ) maxDist = dist;//Delete?
		}
		//cerr << "Min Dist: " << minDist << endl;
		//cerr << "Max Dist: " << maxDist << endl;
		for( int i = 0; i < (int)newDescriptors.rows; i++ ){
		   if( matches[i].distance < 3*minDist ) goodMatches.push_back(matches[i]);
		  }
		/*cerr << "Good Matches: " << goodMatches.size() << endl;

		for(int i = 0; i < goodMatches.size(); i++){
			cerr << "Good Match: " << goodMatches[i].queryIdx << "--" << goodMatches[i].trainIdx << endl;
		}*/

		for (int i = 0; i < (int)goodMatches.size(); i++){
			trackKeypoints.push_back(newKeypoints[goodMatches[i].queryIdx]);
		}


		for (int i = 0; i < (int)goodMatches.size(); i++){
			trackold.push_back(initKeypoints[goodMatches[i].trainIdx]);
		}
		extractor.compute(image,trackKeypoints,trackDescriptors);

		//cerr << "Track Keypoints: " << trackKeypoints.size() << endl;
		//cerr << "Track Descriptors: " << trackDescriptors.rows << endl;

		haveTrack = true;										//Ready to track matches
		//oldImage = image;										//Store current frame for next iteration
	}

	else if(haveTrack && pastImageExists) {							//Track features in following images
		surf.detect(image,newKeypoints);
		extractor.compute(image,newKeypoints,newDescriptors);
		matcher.match(trackDescriptors,newDescriptors,matches);

		for( int i = 0; i < (int)matches.size(); i++ )
		  { double dist = (double)matches[i].distance;					//load element of matches vector
			if( dist < minDist ) minDist = dist;
			if( dist > maxDist ) maxDist = dist;
		  }

		for( int i = 0; i < initDescriptors.rows; i++ )
		  { if( matches[i].distance < 2*minDist)
			 goodMatches.push_back(matches[i]);
		  }

		for( int i=0; i< (int)goodMatches.size();i++){
			cerr << goodMatches[i].trainIdx << endl;
		}

		for (int i = 0; i < (int)matches.size(); i++){
			trackKeypoints.push_back(newKeypoints[matches[i].queryIdx]);
		}
		extractor.compute(image,trackKeypoints,trackDescriptors);
		oldImage = image;
	}

	else {
		cout << "----------->Error: No past image<------------" << endl;
		pastImageExists = false;
	}

	drawKeypoints(image,newKeypoints,image,Scalar(255,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image,trackKeypoints,image,Scalar(0,255,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	drawKeypoints(oldImage,trackold,oldImage,Scalar(0,255,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SURF Features");
	imshow("SURF Features",image);
	waitKey(0);

	namedWindow("Old SURF Features");
		imshow("Old SURF Features",oldImage);
		waitKey(0);
}
