
#include "camFusion.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dataStructures.h"

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, const std::vector<LidarPoint>& lidarPoints,
                         float shrinkFactor, const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    const cv::Mat lidar2Cam = P_rect_xx * R_rect_xx * RT;
    cv::Point pt;

    for (const auto& lPt : lidarPoints) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lPt.x;
        X.at<double>(1, 0) = lPt.y;
        X.at<double>(2, 0) = lPt.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = lidar2Cam * X;

        pt.x = static_cast<int>(Y.at<double>(0, 0) / Y.at<double>(0, 2));  // pixel coordinates
        pt.y = static_cast<int>(Y.at<double>(1, 0) / Y.at<double>(0, 2));

        std::vector<std::vector<BoundingBox>::iterator>
            enclosingBoxes;  // pointers to all bounding boxes which enclose the current Lidar point

        for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = static_cast<int>((*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0);
            smallerBox.y = static_cast<int>((*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0);
            smallerBox.width = static_cast<int>((*it2).roi.width * (1 - shrinkFactor));
            smallerBox.height = static_cast<int>((*it2).roi.height * (1 - shrinkFactor));

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.emplace_back(it2);
            }

        }  // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.emplace_back(lPt);
        }

    }  // eof loop over all Lidar points
}

void show3DObjects(const std::vector<BoundingBox>& boundingBoxes, const cv::Size& worldSize, const cv::Size& imageSize,
                   bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const auto& it1 : boundingBoxes) {
        // create randomized color for current 3D object
        cv::RNG rng(it1.boxID);
        const cv::Scalar currColor(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = std::numeric_limits<int>::max();
        int left = std::numeric_limits<int>::max();
        int bottom = 0;
        int right = 0;

        double xwmin = std::numeric_limits<double>::max();
        double ywmin = std::numeric_limits<double>::max();
        double ywmax = std::numeric_limits<double>::lowest();

        for (const auto& it2 : it1.lidarPoints) {
            // world coordinates
            double xw = it2.x;  // world position in m with x facing forward from sensor
            double yw = it2.y;  // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = static_cast<int>((-xw * imageSize.height / worldSize.height) + imageSize.height);

            int x = static_cast<int>((-yw * imageSize.width / worldSize.width) + imageSize.width / 2);

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        std::stringstream ss;
        ss << "id=" << it1.boxID << ", #pts=" << it1.lidarPoints.size();
        cv::putText(topviewImg, ss.str(), cv::Point2f(left - 250.f, bottom + 50.f), cv::FONT_ITALIC, 2, currColor);

        ss.clear();
        ss.str(std::string());
        ss << std::setprecision(2) << "xmin=" << xwmin << " m, yw=" << ywmax - ywmin;
        cv::putText(topviewImg, ss.str(), cv::Point2f(left - 250.f, bottom + 125.f), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0;  // gap between distance markers
    int nMarkers = static_cast<int>(std::floor(worldSize.height / lineSpacing));
    for (std::size_t i = 0; i < nMarkers; ++i) {
        int y = static_cast<int>((-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height);

        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0);  // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox& boundingBox, const std::vector<cv::KeyPoint>& kptsPrev,
                              const std::vector<cv::KeyPoint>& kptsCurr, const std::vector<cv::DMatch>& kptMatches) {
    for (const auto& match : kptMatches) {
        const auto& currPt = kptsCurr.at(match.trainIdx).pt;
        if (boundingBox.roi.contains(currPt)) {
            boundingBox.kptMatches.emplace_back(match);
        }
    }

    if (boundingBox.kptMatches.empty()) {
        return;
    }

    double sum = 0;
    for (const auto& it : boundingBox.kptMatches) {
        sum += cv::norm(kptsCurr.at(it.trainIdx).pt - kptsPrev.at(it.queryIdx).pt);
    }

    const double mean = sum / boundingBox.kptMatches.size();

    const double ratio = 1.5;
    const double threshold = mean * ratio;

    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();) {
        const auto& kpCurr = kptsCurr.at(it->trainIdx);
        const auto& kpPrev = kptsPrev.at(it->queryIdx);

        if (cv::norm(kpCurr.pt - kpPrev.pt) >= threshold) {
            boundingBox.kptMatches.erase(it);
        } else {
            it++;
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
                      const std::vector<cv::DMatch>& kptMatches, double frameRate, double& TTC, cv::Mat* visImg) {
    // compute distance ratios between all matched keypoints
    std::vector<double> distRatios;       // stores the distance ratios for all keypoints between curr. and prev. frame
    for (const auto& it1 : kptMatches) {  // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        const cv::KeyPoint& kpOuterCurr = kptsCurr.at(it1.trainIdx);
        const cv::KeyPoint& kpOuterPrev = kptsPrev.at(it1.queryIdx);

        for (auto it2 = kptMatches.cbegin() + 1; it2 != kptMatches.cend(); ++it2) {
            // inner kpt.-loop
            double minDist = 100.0;  // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            const cv::KeyPoint& kpInnerCurr = kptsCurr.at(it2->trainIdx);
            const cv::KeyPoint& kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            const double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            const double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {  // avoid division by zero

                const double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }  // eof inner loop over all matched kpts
    }  // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    // double meanDistRatio = std::accumulate(
    //    distRatios.begin(),
    //    distRatios.end(),
    //    0.0) / distRatios.size();

    const double dT = 1 / frameRate;
    // TTC = -dT / (1 - meanDistRatio);

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    std::size_t medIndex = static_cast<std::size_t>(distRatios.size() / 2.0);

    const double medDistRatio = distRatios.size() % 2 == 0
                                    ? (distRatios.at(medIndex - 1) + distRatios.at(medIndex)) / 2.0
                                    : distRatios.at(medIndex);

    TTC = -dT / (1 - medDistRatio);
}

void computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev, const std::vector<LidarPoint>& lidarPointsCurr,
                     double frameRate, double& TTC) {
    // auxiliary variables
    double laneWidth = 4.0;  // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double minXPrev = std::numeric_limits<double>::max();
    double minXCurr = std::numeric_limits<double>::max();

    for (const auto& it : lidarPointsPrev) {
        if (std::abs(it.y) < laneWidth / 2) {
            minXPrev = minXPrev > it.x ? it.x : minXPrev;
        }
    }

    for (const auto& it : lidarPointsCurr) {
        if (std::abs(it.y) < laneWidth / 2) {
            minXCurr = minXCurr > it.x ? it.x : minXCurr;
        }
    }

    // time between two measurements in seconds
    const double dT = 1 / frameRate;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

void matchBoundingBoxes(const std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches,
                        const DataFrame& prevFrame, const DataFrame& currFrame) {
    for (const auto& prevBox : prevFrame.boundingBoxes) {
        std::map<int, int> tmpM;
        for (const auto& currBox : currFrame.boundingBoxes) {
            for (const auto& match : matches) {
                const auto& prevPt = prevFrame.keypoints.at(match.queryIdx).pt;

                if (prevBox.roi.contains(prevPt)) {
                    const auto& currPt = currFrame.keypoints.at(match.trainIdx).pt;

                    if (currBox.roi.contains(currPt)) {
                        if (0 == tmpM.count(currBox.boxID)) {
                            tmpM[currBox.boxID] = 1;
                        } else {
                            tmpM[currBox.boxID]++;
                        }
                    }
                }

            }  // eof iterating all matches
        }  // eof iterating current bounding boxes

        const auto max = *std::max_element(
            tmpM.cbegin(), tmpM.cend(),
            [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) { return p1.second < p2.second; });

        bbBestMatches[prevBox.boxID] = max.first;

    }  // eof iterating previous bounding boxes
}
