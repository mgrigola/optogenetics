#ifndef ILLUMINATEDREGION_H
#define ILLUMINATEDREGION_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <vector>


class IlluminatedRegion
{
private:
    void Copy( const IlluminatedRegion & copySource);
    void Draw_Polygon( cv::Mat& targetImg , int fillType , cv::Scalar drawColor, const std::vector <cv::Point>& bdPts);
    void Draw_Round( cv::Mat& targetImg , int fillType , cv::Scalar drawColor, float eign1, float eign2, float angle, cv::Point2f circCenter);
    void Update_Poly_Bound( void );
    void Get_Eigs( cv::Mat& Q, float & transAngle, float & transEign1, float & transEign2 );
    void Transform_Bounds( std::vector<cv::Point>& bdPts, cv::Mat transformMat);
    void Warp_Bounds( std::vector<cv::Point>& bdPts, cv::Mat transformMat);
    cv::Point2f Transform_Point(cv::Mat& Q, cv::Point2f& pt);

    static int tickFreq;

public:

    enum SHAPE
    {
        SHAPE_CIRCLE,
        SHAPE_SQUARE_0,
        SHAPE_SQUARE_45,
        SHAPE_RECT_0,
        SHAPE_RECT_23,
        SHAPE_RECT_45,
        SHAPE_RECT_68,
        SHAPE_RECT_90,
        SHAPE_RECT_113,
        SHAPE_RECT_135,
        SHAPE_RECT_158,
        SHAPE_DIAMOND_0,
        SHAPE_DIAMOND_45,
        SHAPE_DIAMOND_90,
        SHAPE_DIAMOND_135,
        SHAPE_MAX = SHAPE_DIAMOND_135
    };

    int regionRad, regionShape, regionFill;
    int64 offTick, onTick;
    bool pulseOn, isActive, beingEdited;
    int dutyCycle, pulseFreq, phaseShift;
    float startWgt;

    cv::Point2f regionCenter;
    cv::Scalar regionColor;
    cv::Mat transMat;
    std::vector <cv::Point> boundPts, boundPts2;
    std::vector <cv::Point2f> boundOffsets;
    std::vector <cv::Point2f> keyPts;

    IlluminatedRegion( void );
    IlluminatedRegion(const IlluminatedRegion& other);
    ~IlluminatedRegion(void);
    IlluminatedRegion& operator = (const IlluminatedRegion& copySource);
    void Update_Blink_State(int64 tickCount );

    void Draw_Region(cv::Mat &targetImg, int fillType = 0, cv::Scalar drawColor = cv::Scalar(-1,-1,-1));
    void Draw_Transformed_Region(cv::Mat& targetImg, int fillType = 0, cv::Scalar drawColor = cv::Scalar(-1,-1,-1));
    void Draw_Key_Points( cv::Mat& targetImage );
    static void Set_Transf( cv::Mat Q );
    static cv::Mat Get_Transf( void );

    static cv::Mat camProjTransf;

};


#endif // ILLUMINATEDREGION_H
