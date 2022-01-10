#ifndef dataStructures_h
#define dataStructures_h

#include <deque>
#include <map>
#include <opencv2/core.hpp>
#include <vector>


template <class T, std::size_t capacity>
class Ringbuffer {
    std::deque<T> buffer;

   public:
    void push_back(const T& data) {
        buffer.push_back(data);
        if (buffer.size() > capacity) {
            buffer.pop_front();
        }
    }

    bool empty() const {
        return buffer.empty();
    }

    std::size_t size() const {
        return buffer.size();
    }

    typename std::deque<T>::iterator begin() {
        return buffer.begin();
    }

    typename std::deque<T>::iterator end() {
        return buffer.end();
    }

    T& back() {
        return buffer.back();
    }
};

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};

#endif /* dataStructures_h */
