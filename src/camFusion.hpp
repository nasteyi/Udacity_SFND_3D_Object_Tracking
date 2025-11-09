
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>

#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "dataStructures.h"

void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, const std::vector<LidarPoint>& lidarPoints,
                         float shrinkFactor, const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT);

void clusterKptMatchesWithROI(BoundingBox& boundingBox, const std::vector<cv::KeyPoint>& kptsPrev,
                              const std::vector<cv::KeyPoint>& kptsCurr, const std::vector<cv::DMatch>& kptMatches);

void matchBoundingBoxes(const std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches,
                        const DataFrame& prevFrame, const DataFrame& currFrame);

void show3DObjects(const std::vector<BoundingBox>& boundingBoxes, const cv::Size& worldSize, const cv::Size& imageSize,
                   bool bWait = true);

void computeTTCCamera(const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
                      const std::vector<cv::DMatch>& kptMatches, double frameRate, double& TTC,
                      cv::Mat* visImg = nullptr);

void computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev, const std::vector<LidarPoint>& lidarPointsCurr,
                     double frameRate, double& TTC);

#endif /* camFusion_hpp */
