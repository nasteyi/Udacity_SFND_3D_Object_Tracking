#ifndef dataStructures_h
#define dataStructures_h

#include <map>
#include <opencv2/core.hpp>
#include <vector>

struct LidarPoint {  // single lidar point in space
    double x{0};
    double y{0};
    double z{0};
    double r{0};  // x,y,z in [m], r is point reflectivity
};

struct BoundingBox {  // bounding box around a classified object (contains both 2D and 3D data)

    int boxID{-1};    // unique identifier for this bounding box
    int trackID{-1};  // unique identifier for the track to which this bounding box belongs

    cv::Rect roi;          // 2D region-of-interest in image coordinates
    int classID{-1};       // ID based on class file provided to YOLO framework
    double confidence{0};  // classification trust

    std::vector<LidarPoint> lidarPoints;  // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints;  // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches;   // keypoint matches enclosed by 2D roi
};

struct DataFrame {  // represents the available sensor information at the same time instance

    cv::Mat cameraImg;  // camera image

    std::vector<cv::KeyPoint> keypoints;  // 2D keypoints within camera image
    cv::Mat descriptors;                  // keypoint descriptors
    std::vector<cv::DMatch> kptMatches;   // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes;  // ROI around detected objects in 2D image coordinates
    std::map<int, int> bbMatches;            // bounding box matches between previous and current frame
};

#endif /* dataStructures_h */
