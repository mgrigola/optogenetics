#include "IlluminatedRegion.h"

int IlluminatedRegion::tickFreq = static_cast<int>(cv::getTickFrequency());
cv::Mat IlluminatedRegion::camProjTransf = cv::Mat::eye(Size(3,3), CV_32F);

IlluminatedRegion::IlluminatedRegion(void) :
    regionCenter(cv::Point2f(0,0)),
    regionColor(cv::Scalar(255,0,0)),
    regionRad(50),
    dutyCycle(50),
    isActive(false),
    pulseFreq(100),
    phaseShift(0),
    transMat(cv::Mat::eye(cv::Size(3,3), CV_32F)),
    boundPts(4, cv::Point(0,0)),
    regionFill(1),
    startWgt(0),
    beingEdited(true),
    boundOffsets(4, cv::Point2f(0,0)),
    boundPts2(4, cv::Point(0,0))
{
}


IlluminatedRegion::IlluminatedRegion(const IlluminatedRegion& other)
{
    this->Copy(other);
}

IlluminatedRegion::~IlluminatedRegion(void)
{
}


IlluminatedRegion& IlluminatedRegion::operator=(const IlluminatedRegion& other)
{
    this->Copy(other);
    return *this;
}


void IlluminatedRegion::Copy(const IlluminatedRegion& other)
{
    regionRad = other.regionRad;
    regionShape = other.regionShape;
    regionFill = other.regionFill;
    offTick = other .offTick;
    onTick = other.onTick;
    pulseOn = other.pulseOn;
    isActive = other.isActive;
    beingEdited = other.beingEdited;
    dutyCycle = other.dutyCycle;
    pulseFreq = other.pulseFreq;
    phaseShift = other.phaseShift;
    startWgt = other.startWgt;
    regionCenter = other.regionCenter;
    regionColor = other.regionColor;
    other.transMat.copyTo(transMat);
    boundPts = other.boundPts;
    boundPts2 = other.boundPts2;
    boundOffsets = other.boundOffsets;
    keyPts = other.keyPts;
}


void IlluminatedRegion::Update_Blink_State(int64 tickCount)
{
    if (dutyCycle == 100)
    {
        pulseOn = true;
        return;
    }

    if (dutyCycle == 0)
    {
        pulseOn = false;
        return;
    }

    if (pulseOn)
    {
        if (tickCount < offTick)  return;
        else
        {
            pulseOn = false;
            onTick = offTick + dutyCycle*tickFreq/pulseFreq;
        }
    }
    else
    {
        if (tickCount < onTick)  return;
        else
        {
            pulseOn = true;
            offTick = onTick + (100-dutyCycle)*tickFreq/pulseFreq;
        }
    }
}



void IlluminatedRegion::Get_Eigs(cv::Mat& Q,  float& transAngle, float& singVal1, float& singVal2)
{
    cv::Mat sVals, sVecs, garbage;
    cv::SVD::compute(Q(cv::Rect(0,0,2,2)), sVals, sVecs, garbage);


    singVal1 = sVals.at<float>(0);
    singVal2 = sVals.at<float>(1);
    transAngle = 90 + 180.0 / CV_PI * atanf( sVecs.at<float>(0,0) / (.00001 + sVecs.at<float>(1,0)) );
}



void IlluminatedRegion::Update_Poly_Bound( void )
{
    float cAng, sAng;
    cv::Point2f a,b;

    switch ( regionShape )
    {

    case SHAPE_SQUARE_0:
        a = .82*regionRad*cv::Point2f( 1 , 0 );
        b = .82*regionRad*cv::Point2f( 0 , 1 );
        break;

    case SHAPE_SQUARE_45:
        a = .6*regionRad*cv::Point2f( 1 , -1 );
        b = .6*regionRad*cv::Point2f( 1 ,  1 );
        break;

    case SHAPE_RECT_0:
        cAng = 1;
        sAng = 0;
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_23:
        cAng = cosf( .125*CV_PI );
        sAng = sinf( .125*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_45:
        cAng = cosf( .25*CV_PI );
        sAng = sinf( .25*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_68:
        cAng = cosf( .375*CV_PI );
        sAng = sinf( .375*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_90:
        cAng = 0;
        sAng = 1;
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_113:
        cAng = cosf( .625*CV_PI );
        sAng = sinf( .625*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_135:
        cAng = cosf( .75*CV_PI );
        sAng = sinf( .75*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_RECT_158:
        cAng = cosf( .875*CV_PI );
        sAng = sinf( .875*CV_PI );
        a = regionRad*cv::Point2f( cAng , -sAng );
        b = regionRad*cv::Point2f( .5*sAng,  .5*cAng );
        break;

    case SHAPE_DIAMOND_0:
        cAng = 1;
        sAng = 0;
        a = .75*regionRad*cv::Point2f( .5*cAng - sAng  , cAng + .5*sAng);
        b = .75*regionRad*cv::Point2f( -.5*cAng - sAng , cAng - .5*sAng);
        break;

    case SHAPE_DIAMOND_45:
        cAng = cosf(.25*CV_PI);
        sAng = sinf(.25*CV_PI);
        a = .75*regionRad*cv::Point2f( .5*cAng - sAng  , cAng + .5*sAng);
        b = .75*regionRad*cv::Point2f( -.5*cAng - sAng , cAng - .5*sAng);
        break;

    case SHAPE_DIAMOND_90:
        cAng = 0;
        sAng = 1;
        a = .75*regionRad*cv::Point2f( .5*cAng - sAng  , cAng + .5*sAng);
        b = .75*regionRad*cv::Point2f( -.5*cAng - sAng , cAng - .5*sAng);
        break;

    case SHAPE_DIAMOND_135:
        cAng = cosf(.75*CV_PI);
        sAng = sinf(.75*CV_PI);
        a = .75*regionRad*cv::Point2f( .5*cAng - sAng  , cAng + .5*sAng);
        b = .75*regionRad*cv::Point2f( -.5*cAng - sAng , cAng - .5*sAng);
        break;

    default:
        return;
    }

    boundOffsets[0] =  a + b;
    boundOffsets[1] = -a + b;
    boundOffsets[2] = -a - b;
    boundOffsets[3] =  a - b;
}



void IlluminatedRegion::Draw_Polygon(cv::Mat& targetImg, int fillType, cv::Scalar drawColor, const std::vector <cv::Point>& bdPts)
{
    if (fillType == CV_FILLED)
        cv::fillConvexPoly(targetImg , &bdPts.front(), bdPts.size(), drawColor, CV_AA);
    else
        for (uchar ptNo=0; ptNo<bdPts.size(); ptNo++)
            cv::line( targetImg , bdPts[ptNo] , bdPts[(ptNo+1)%bdPts.size()] , drawColor , fillType , CV_AA );
}


void IlluminatedRegion::Draw_Round( cv::Mat& targetImg , int fillType , cv::Scalar drawColor, float angle, float eign1, float eign2, cv::Point2f ptCenter)
{
    cv::ellipse(targetImg, ptCenter, Size(regionRad*eign1, regionRad*eign2), angle, 0, 360, drawColor, fillType, CV_AA);
}


void IlluminatedRegion::Draw_Region(cv::Mat& targetImg, int fillType, cv::Scalar drawColor)
{
    if (drawColor[0] == -1)
        drawColor = regionColor;

    if (fillType == 0)
        fillType = regionFill;

    if (regionShape == SHAPE_CIRCLE)
    {
//		Get_Eigs(transMat, boundOffsets[2].x, boundOffsets[1].x, boundOffsets[1].y);
//		Draw_Round( targetImg, fillType, drawColor, boundOffsets[2].x, boundOffsets[1].x, boundOffsets[1].y, regionCenter);
        Draw_Round( targetImg, fillType, drawColor, 0, 1, 1, regionCenter);
    }
    else
    {
        if (beingEdited)
            Update_Poly_Bound();

        Warp_Bounds( boundPts, transMat );
        Draw_Polygon( targetImg, fillType, drawColor, boundPts);
    }
}


void IlluminatedRegion::Draw_Key_Points(cv::Mat& targetImage)
{
    //### remove this iterator crap. hate it!
    for ( std::vector<cv::Point2f>::iterator kk = keyPts.begin() ; kk != keyPts.end() ; ++kk )
        cv::circle(targetImage, *kk, 1, cv::Scalar(255,255,255), CV_FILLED);

//    for (cv::Point& keyPt : keyPts)
//        cv::circle(targetImage, keyPt, 1, cv::Scalar(255,255,255), CV_FILLED);
}


void IlluminatedRegion::Warp_Bounds( std::vector<cv::Point>& bdPts, cv::Mat transformMat)
{
    const float* Q0 = transformMat.ptr<float>(0);
    const float* Q1 = transformMat.ptr<float>(1);

    for (int ptNo=0; ptNo<boundOffsets.size(); ptNo++)
        bdPts[ptNo] = Point(  Q0[0]*boundOffsets[ptNo].x + Q0[1]*boundOffsets[ptNo].y + regionCenter.x ,  Q1[0]*boundOffsets[ptNo].x + Q1[1]*boundOffsets[ptNo].y + regionCenter.y );
}



void IlluminatedRegion::Transform_Bounds( vector<Point>& bdPts, Mat transformMat)
{
    const float* Q0 = transformMat.ptr<float>(0);
    const float* Q1 = transformMat.ptr<float>(1);

    for (int ptNo=0; ptNo<boundOffsets.size(); ptNo++)
        bdPts[ptNo] = cv::Point(  Q0[0]*float(boundPts[ptNo].x) +  Q0[1]*float(boundPts[ptNo].y) + Q0[2] ,  Q1[0]*float(boundPts[ptNo].x) +  Q1[1]*float(boundPts[ptNo].y) + Q1[2]);

//	for (int kk=0; kk<boundOffsets.size(); kk++)
//		bdPts[kk] = Point(  Q0[0]*(regionCenter.x+boundOffsets[kk].x) + Q0[1]*(regionCenter.y+boundOffsets[kk].y) + Q0[2] ,  Q1[0]*(regionCenter.x + boundOffsets[kk].x) + Q1[1]*(regionCenter.y + boundOffsets[kk].y) + Q1[2] );
}



void IlluminatedRegion::Draw_Transformed_Region(cv::Mat& targetImg, int fillType, cv::Scalar drawColor)
{
    if (drawColor[0] == -1)
        drawColor = regionColor;

    if (fillType == 0)
        fillType = regionFill;

    if (regionShape == SHAPE_CIRCLE)
    {
        Get_Eigs(transMat*camProjTransf, boundOffsets[2].y, boundOffsets[3].x, boundOffsets[3].y);
        Draw_Round(targetImg, fillType, drawColor, boundOffsets[2].y, boundOffsets[3].x, boundOffsets[3].y, Transform_Point(camProjTransf, regionCenter));
    }
    else
    {
        Transform_Bounds( boundPts2, camProjTransf );
        Draw_Polygon( targetImg, fillType, drawColor, boundPts2);
    }
}



void IlluminatedRegion::Set_Transf(cv::Mat& Q)
{
    Q.copyTo(camProjTransf);}


cv::Mat IlluminatedRegion::Get_Transf( void )
{
    return camProjTransf;
}


//###pretty sure there's a built in function that does this. might not have been 5  yrs ago.
cv::Point2f IlluminatedRegion::Transform_Point(cv::Mat& Q, cv::Point2f& pt)
{
    const float* Q0 = Q.ptr<float>(0);
    const float* Q1 = Q.ptr<float>(1);

    return cv::Point2f( Q0[0]*pt.x + Q0[1]*pt.y + Q0[2] , Q1[0]*pt.x + Q1[1]*pt.y + Q1[2] );
}
