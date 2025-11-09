
#include "lidarData.hpp"

#include <algorithm>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// remove Lidar points based on min. and max distance in X, Y and Z
void cropLidarPoints(std::vector<LidarPoint>& lidarPoints, float minX, float maxX, float maxY, float minZ, float maxZ,
                     float minR) {
    std::vector<LidarPoint> newLidarPts;
    for (const auto& lPt : lidarPoints) {
        // Check if Lidar point is outside of boundaries
        if (lPt.x >= minX && lPt.x <= maxX && lPt.z >= minZ && lPt.z <= maxZ && lPt.z <= 0.0 &&
            std::abs(lPt.y) <= maxY && lPt.r >= minR) {
            newLidarPts.emplace_back(lPt);
        }
    }

    lidarPoints = newLidarPts;
}

// Load Lidar points from a given location and store them in a vector
void loadLidarFromFile(std::vector<LidarPoint>& lidarPoints, const std::string& filename) {
    // load point cloud
    FILE* stream = fopen(filename.c_str(), "rb");
    if (stream == NULL) {
        return;
    }

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    unsigned long long num = 1000000;
    float* data = (float*)malloc(num * sizeof(float));

    // pointers
    float* px = data + 0;
    float* py = data + 1;
    float* pz = data + 2;
    float* pr = data + 3;

    num = fread(data, sizeof(float), num, stream) / 4;

    for (std::int32_t i = 0; i < num; ++i) {
        LidarPoint lpt;

        lpt.x = *px;
        lpt.y = *py;
        lpt.z = *pz;
        lpt.r = *pr;

        lidarPoints.emplace_back(lpt);

        px += 4;
        py += 4;
        pz += 4;
        pr += 4;
    }

    free(data);
    data = nullptr;
    fclose(stream);
}

void showLidarTopview(const std::vector<LidarPoint>& lidarPoints, const cv::Size& worldSize, const cv::Size& imageSize,
                      bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    for (const auto& lPt : lidarPoints) {
        const double xw = lPt.x;  // world position in m with x facing forward from sensor
        const double yw = lPt.y;  // world position in m with y facing left from sensor

        int y = static_cast<int>((-xw * imageSize.height / worldSize.height) + imageSize.height);
        int x = static_cast<int>((-yw * imageSize.height / worldSize.height) + imageSize.width / 2);

        cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }

    // plot distance markers
    float lineSpacing = 2.0;  // gap between distance markers
    int nMarkers = static_cast<int>(std::floor(worldSize.height / lineSpacing));
    for (std::size_t i = 0; i < nMarkers; ++i) {
        int y = static_cast<int>((-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height);

        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    if (bWait) {
        cv::waitKey(0);  // wait for key to be pressed
    }
}

void showLidarImgOverlay(cv::Mat& img, std::vector<LidarPoint>& lidarPoints, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx,
                         cv::Mat& RT, cv::Mat* extVisImg) {
    // init image for visualization
    cv::Mat visImg = nullptr == extVisImg ? img.clone() : *extVisImg;
    cv::Mat overlay = visImg.clone();

    // find max. x-value
    double maxVal = 0.0;
    for (const auto& lPt : lidarPoints) {
        maxVal = maxVal < lPt.x ? lPt.x : maxVal;
    }

    auto maxX = std::max_element(lidarPoints.begin(), lidarPoints.end(),
                                 [](const LidarPoint& pt1, const LidarPoint& pt2) { return pt1.x > pt2.x; });

    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    const cv::Mat lidar2Cam = P_rect_xx * R_rect_xx * RT;
    cv::Point pt;

    for (const auto& lPt : lidarPoints) {
        X.at<double>(0, 0) = lPt.x;
        X.at<double>(1, 0) = lPt.y;
        X.at<double>(2, 0) = lPt.z;
        X.at<double>(3, 0) = 1;

        Y = lidar2Cam * X;

        pt.x = static_cast<int>(Y.at<double>(0, 0) / Y.at<double>(0, 2));
        pt.y = static_cast<int>(Y.at<double>(1, 0) / Y.at<double>(0, 2));

        const double val = lPt.x;
        const double t = std::abs((val - maxVal) / maxVal);
        int red = std::min(255, (int)(255 * t));
        int green = std::min(255, (int)(255 * (1 - t)));
        cv::circle(overlay, pt, 5, cv::Scalar(0, green, red), -1);
    }

    const double opacity = 0.6;
    cv::addWeighted(overlay, opacity, visImg, 1 - opacity, 0, visImg);

    // return augmented image or wait if no image has been provided
    if (extVisImg == nullptr) {
        std::string windowName = "LiDAR data on image overlay";
        cv::namedWindow(windowName, 3);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);  // wait for key to be pressed
    } else {
        *extVisImg = visImg.clone();
    }
}