
#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes;  // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        }  // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    }  // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x;  // world position in m with x facing forward from sensor
            float yw = (*it2).y;  // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

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
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.4f m, yw=%2.4f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0;  // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0);  // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
    std::vector<std::pair<double, cv::DMatch>> distances;

    for (const auto &match : kptMatches) {
        const auto currKpt = kptsCurr[match.trainIdx];
        const auto prevKpt = kptsPrev[match.queryIdx];

        if (boundingBox.roi.contains(currKpt.pt)) {
            const auto distance = cv::norm(currKpt.pt - prevKpt.pt);
            distances.push_back({distance, match});
        }
    }
    std::sort(distances.begin(), distances.end(), [](const std::pair<double, cv::DMatch> &a, const std::pair<double, cv::DMatch> &b) { return a.first < b.first; });
    // again use IQR for outlier detection

    const auto Q1 = distances[distances.size() / 4].first;
    const auto Q3 = distances[distances.size() * 3 / 4].first;
    const auto IQR = Q3 - Q1;
    for (const auto &match : distances) {
        const auto distance = match.first;
        if (distance < Q1 + 1.5 * IQR) {
            boundingBox.kptMatches.push_back(match.second);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
    // store ratios of keypoint distances to
    std::vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++) {
        const auto kpOuterCurr = kptsCurr.at(it1->trainIdx);
        const auto kpOuterPrev = kptsPrev.at(it1->queryIdx);
        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2) {
            const auto kpInnerCurr = kptsCurr.at(it2->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
            const auto kpInnerPrev = kptsPrev.at(it2->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

            // Use cv::norm to calculate the current and previous Euclidean distances between each keypoint in the pair
            const auto distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            const auto distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            constexpr double minDist = 100.0;  // Threshold the calculated distRatios by requiring a minimum current distance between keypoints

            // Avoid division by zero and apply the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if the vector of distRatios is not empty
    if (distRatios.empty()) {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    if (lidarPointsPrev.empty() || lidarPointsCurr.empty()) {
        return;
    }

    const auto compLidarPoints = [](const LidarPoint &a, const LidarPoint &b) { return a.x < b.x; };

    // find closest distance to Lidar points
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compLidarPoints);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compLidarPoints);

    // first and third quartiles
    const auto prevQ1 = lidarPointsPrev[lidarPointsPrev.size() / 4].x;
    const auto prevQ3 = lidarPointsPrev[lidarPointsPrev.size() * 3 / 4].x;
    const auto currQ1 = lidarPointsCurr[lidarPointsCurr.size() / 4].x;
    const auto currQ3 = lidarPointsCurr[lidarPointsCurr.size() * 3 / 4].x;

    // compute interquartile range: https://en.wikipedia.org/wiki/Interquartile_range
    const auto prevIQR = prevQ3 - prevQ1;
    const auto currIQR = currQ3 - currQ1;

    const auto compLidarPointsAndDouble = [](const LidarPoint &a, const double b) { return a.x < b; };

    const auto minXPrev = std::lower_bound(lidarPointsPrev.begin(), lidarPointsPrev.end(), prevQ1 - 1.5 * prevIQR, compLidarPointsAndDouble)->x;
    const auto minXCurr = std::lower_bound(lidarPointsCurr.begin(), lidarPointsCurr.end(), currQ1 - 1.5 * currIQR, compLidarPointsAndDouble)->x;

    std::cout << "Lidar min x prev/curr: " << minXPrev << " " << minXCurr << std::endl;

    // compute TTC from both measurements
    TTC = minXCurr / (frameRate * (minXPrev - minXCurr));
    std::cout << "Lidar TTC: " << TTC << std::endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    // std::multimap not really useful to count number of occurences of values
    std::map<int, std::map<int, int>> matchedBoundingBoxIds;
    for (const auto &match : matches) {
        const auto prevKeypoint = prevFrame.keypoints[match.queryIdx];
        const auto currKeypoint = currFrame.keypoints[match.trainIdx];

        std::vector<int> prevMatchedBoundingBoxIds;

        for (const auto &boundingBox : prevFrame.boundingBoxes) {
            if (boundingBox.roi.contains(prevKeypoint.pt)) {
                prevMatchedBoundingBoxIds.push_back(boundingBox.boxID);
            }
        }

        for (const auto &boundingBox : currFrame.boundingBoxes) {
            if (boundingBox.roi.contains(currKeypoint.pt)) {
                for (const auto prevID : prevMatchedBoundingBoxIds) {
                    ++matchedBoundingBoxIds[prevID][boundingBox.boxID];
                }
            }
        }
    }
    for (const auto &matchCounts : matchedBoundingBoxIds) {
        const auto prevId = matchCounts.first;
        int bestMatch{};
        int bestCount{0};
        for (const auto currIdCounts : matchCounts.second) {
            if (currIdCounts.second > bestCount) {
                bestMatch = currIdCounts.first;
                bestCount = currIdCounts.second;
            }
        }
        bbBestMatches[prevId] = bestMatch;
    }
}
