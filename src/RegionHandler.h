#ifndef REGIONHANDLER_H
#define REGIONHANDLER_H

#include "IlluminatedRegion.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

class RegionHandler
{
private:
    std::vector <cv::Point2f> trackPts;
    std::vector <cv::Point2f> newPts;
    std::vector<float> err;
    std::vector<uchar> status;
    const cv::Mat noTransform;
    const cv::Size winSize;
    const cv::TermCriteria termCrit;
    int trackPtNo;

    bool Update_Track_Region (IlluminatedRegion& trackRegion);
    bool Handle_Track_Fail(const cv::Mat& img, IlluminatedRegion& failRegion);
    bool Initialize_Region(const cv::Mat& grayImage, IlluminatedRegion& initRegion);
    void Draw_Track_Points(cv::Mat& camImage);
    void Assemble_Track_Vector( void );

public:
    static int tickFreq;

    std::vector <IlluminatedRegion> regionList;

    int trackObjectNo;
    bool uniformTiming;
    bool showTrackPts, blackWhenOff, blackWhenOffPrecedence;

    RegionHandler(void);
    ~RegionHandler(void);
    void Reset_Timing(IlluminatedRegion& irMaster, int regionNo=99999);
    void Draw_Regions_On_Projector(cv::Mat& projImage );
    void Draw_Regions_On_Camera(cv::Mat& camImage, bool trackingOn=true );
    void Track(const cv::Mat& oldImg, const cv::Mat& newImg );
    void Clear_Tracks( void );
    void Delete_Last_Rgn( void );
    void Save_Text_Track(std::ofstream& cellLocStrm, bool saveKeyPts);
    void Refresh_Track_Pts(cv::Mat& targetImage);
    cv::Mat LS_Rot_Stretch_Trans(std::vector<cv::Point2f>& ptOrig, std::vector<cv::Point2f>& ptTrans, std::vector<float> ptWeights);
    cv::Mat LS_Affine_Trans(std::vector<cv::Point2f>& ptOrig, std::vector<cv::Point2f>& ptTrans, std::vector<float> ptWeights);
    cv::Mat LS_Rot_Trans(std::vector<cv::Point2f>& ptOrig, std::vector<cv::Point2f>& ptTrans, std::vector<float> ptWeights);
    void Add_Region(const cv::Mat& grayImage, IlluminatedRegion& initRegion);

 };


#endif // REGIONHANDLER_H
