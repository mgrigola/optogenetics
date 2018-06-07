#include "RegionHandler.h"

int RegionHandler::tickFreq = static_cast<int>(getTickFrequency());


RegionHandler::RegionHandler(void) :
    winSize(Size(19,19)),
    termCrit(cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,30,0.001)),
    blackWhenOff(true),
    blackWhenOffPrecedence(true),
    trackObjectNo(0),
    uniformTiming(true),
    showTrackPts(true)
{
//	trackPts.reserve(1024);
//	regionList.reserve(512);
}


RegionHandler::~RegionHandler(void)
{
    Clear_Tracks();
}


void RegionHandler::Add_Region( const cv::Mat& grayImage , IlluminatedRegion& addRegion )
{
    regionList.push_back(addRegion);
    if ( !Initialize_Region( grayImage, regionList.back() ) )
        regionList.pop_back();

    regionList.back().beingEdited = false;
}


bool RegionHandler::Initialize_Region(const cv::Mat& grayImage , IlluminatedRegion& initRegion )
{
    cv::Mat selectMask = cv::Mat::zeros(grayImage.rows, grayImage.cols, CV_8U);
    initRegion.Draw_Region(selectMask, CV_FILLED, cv::Scalar(1,1,1));

    //finds sharp corners / distinctive features in the image that we'll try to follow frame-to-frame. This is the initial seed
    cv::goodFeaturesToTrack(grayImage, initRegion.keyPts, 128, 0.01, 3, selectMask, 5, true, -.5);

    initRegion.startWgt = 0;

    if (initRegion.keyPts.size() == 0 )
    {
        initRegion.isActive = false;
        return false;
    }
    else
    {
        initRegion.isActive = true;
        Reset_Timing( initRegion , 0 );
    }

    return true;
}


void RegionHandler::Assemble_Track_Vector( void )
{
    trackPts.clear();

    //combine all the tracking points from any regions into one vector to update all in 1 call to optic flow
    for (IlluminatedRegion region : regionList)
        for (cv::Point2f keyPt : keyPts)
            trackPts.push_back(keyPt);
}


void RegionHandler::Track( const cv::Mat& oldImg , const cv::Mat& newImg )
{

    Assemble_Track_Vector();
    if ( trackPts.size() == 0 )
        return;

    calcOpticalFlowPyrLK( oldImg, newImg, trackPts, newPts, status, err, winSize, 3, termCrit, CV_LKFLOW_GET_MIN_EIGENVALS, .001 );

    trackPtNo = 0;  //### why?
    for (IlluminatedRegion region : regionList)
    {
        if (!region.isActive)
            continue;

        if (!Update_Track_Region(region) )
            Handle_Track_Fail(newImg, region);
    }
}

/**********************************************************************************************************************************************************************************************/

bool RegionHandler::Handle_Track_Fail( const cv::Mat& img, IlluminatedRegion& failRegion )
{
    cv::Point2f fCent = failRegion.regionCenter;
    int fRad = failRegion.regionRad;

    // if region is close to edge ro realyl tiny, just give up on this region
    if ( fCent.x<4 || fCent.y<4 || fCent.x>(img.cols-5) || fCent.y>(img.rows-5) || tempRad < 2)
    {
        failRegion.isActive = false;
        return false;
    }

    // make adjustments to region to try and keep the region on screen - shift center away from edge and shrink radius accordingly
    if (fCent.x-fRad < 0)           { fCent.x += 2;  fRad = (fCent.x-1)/2; }
    if (fCent.y-fRad < 0)           { fCent.y += 2;  fRad = (fCent.y-1)/2; }
    if (fCent.x+fRad > img.cols)    { fCent.x -= 2;  fRad = (img.cols - fCent.x - 2)/2; }
    if (fCent.y+fRad > img.rows)    { fCent.x -= 2;  fRad = (img.rows - fCent.y - 2)/2; }

    failRegion.regionRad = fRad;
    failRegion.regionCenter = fCent;

    if ( !Initialize_Region(img, failRegion) )
    {
        failRegion.isActive = false;
        failRegion.transMat = cv::Mat::eye(Size(3,3), CV_32F);
        return false;
    }

    return true;
}


bool RegionHandler::Update_Track_Region(IlluminatedRegion& trackRegion)
{
    float totWeight = 0;
    int tempRad = trackRegion.regionRad;
    float radSq = pow(tempRad,2);
    int endIdx = trackPtNo + trackRegion.keyPts.size();

    std::vector<float> ptWgts;
    std::vector<cv::Point2f> ptOrig;
    std::vector<cv::Point2f> ptTrans;

    trackRegion.keyPts.clear();

    while ( trackPtNo < endIdx )
    {
        if ( status[trackPtNo] )
        {
            float distSq  = pow(newPts[trackPtNo].x - trackRegion.regionCenter.x, 2) + pow(newPts[trackPtNo].y - trackRegion.regionCenter.y,2);

            float tempWeight = 2*radSq - distSq;
            tempWeight *= err[trackPtNo]*1000;

            if ( tempWeight > radSq )
            {
                ptWgts.push_back(tempWeight);
                ptOrig.push_back(trackPts[trackPtNo] - trackRegion.regionCenter);
                ptTrans.push_back(newPts[trackPtNo] - trackRegion.regionCenter);

                totWeight += tempWeight;

                trackRegion.keyPts.push_back( newPts[trackPtNo] );
            }
        }

        ++trackPtNo;
    }

    if ( totWeight < .5*trackRegion.startWgt )  // track region fail
    {
        if (trackRegion.startWgt == 0 )
            trackRegion.startWgt = totWeight;
        else
            return false;
    }

    Mat Q = LS_Rot_Trans(ptOrig, ptTrans, ptWgts);

    trackRegion.transMat(Rect(0,0,2,2)) = Q(Rect(0,0,2,2))*trackRegion.transMat(Rect(0,0,2,2));
    trackRegion.transMat(Rect(0,0,2,2)) /= sqrt(determinant(trackRegion.transMat(Rect(0,0,2,2))));
    trackRegion.regionCenter += Point2f(Q.at<float>(0,2) , Q.at<float>(1,2));

    return true;
}

/**********************************************************************************************************************************************************************************************/

void RegionHandler::Clear_Tracks( void )
{
    trackPts.clear();
    status.clear();
    err.clear();
    regionList.clear();
}

/**********************************************************************************************************************************************************************************************/

void RegionHandler::Delete_Last_Rgn( void )
{
    regionList.pop_back();
}

/**********************************************************************************************************************************************************************************************/

void RegionHandler::Save_Text_Track( ofstream &cellLocStrm, bool saveKeyPts)
{
    for (int kk = 0; kk< regionList.size(); kk++)
    {
        if (regionList[kk].isActive)
        {
            if (saveKeyPts)
                for (int jj=0; jj<regionList[kk].keyPts.size(); jj++)
                    cellLocStrm << regionList[kk].keyPts[jj].x << " " << regionList[kk].keyPts[jj].y << " "<< 1 << " ";
            else
                cellLocStrm << regionList[kk].regionCenter.x << " " << regionList[kk].regionCenter.y << " "<< regionList[kk].regionRad << " ";
        }
        else
            cellLocStrm << "-1 -1 -1 ";
    }


    cellLocStrm << endl;
}

/**********************************************************************************************************************************************************************************************/

void RegionHandler::Refresh_Track_Pts(cv::Mat& targetImage)
{
    trackPts.clear();
    status.clear();
    err.clear();
    for (IlluminatedRegion region : regionList)
        if (region.isActive)  Initialize_Region(targetImage, region);
}


cv::Mat RegionHandler::LS_Rot_Stretch_Trans( std::vector<cv::Point2f> ptOrig, std::vector<cv::Point2f> ptTrans, std::vector<float> ptWeights )
{
    // Q = [  a   b   e  ]
    //     [ -b   c   f  ]
    // A = [x  y  1  0  0]  ->  X = [ X ]
    //   = [0 -x  0  y  1]  ->      [ Y ]

    Mat linQ;
    Mat Q( Size(3,2) , CV_32F);
    Mat A( Size( 5 , 2*ptOrig.size() ) , CV_32F);
    Mat X( Size( 1 , 2*ptOrig.size() ) , CV_32F);

    for (int kk = 0; kk<ptOrig.size(); kk++)
    {
        float* Akk = A.ptr<float>(2*kk);
        float* Xkk = X.ptr<float>(2*kk);
        Akk[0] = ptOrig[kk].x * ptWeights[kk] + 3*ptWeights[kk];
        Akk[1] = ptOrig[kk].y * ptWeights[kk];
        Akk[2] = ptWeights[kk];
        Akk[3] = 0;
        Akk[4] = 0;
        Xkk[0] = ptTrans[kk].x * ptWeights[kk] + 3*ptWeights[kk];

        Akk = A.ptr<float>(2*kk+1);
        Xkk = X.ptr<float>(2*kk+1);
        Akk[0] = 0;
        Akk[1] = -ptOrig[kk].x * ptWeights[kk];
        Akk[2] = 0;
        Akk[3] = ptOrig[kk].y * ptWeights[kk] + 4*ptWeights[kk];
        Akk[4] = ptWeights[kk];
        Xkk[0] = ptTrans[kk].y * ptWeights[kk] + 4*ptWeights[kk];
    }
    linQ = ((((A.t()*A).inv(cv::DECOMP_LU))*A.t())*X).t();

    float* lQ = linQ.ptr<float>(0);
    float* qQ = Q.ptr<float>(0);
    qQ[0] = lQ[0];
    qQ[1] = lQ[1];
    qQ[2] = lQ[2];
    qQ = Q.ptr<float>(1);
    qQ[0] = -lQ[1];
    qQ[1] = lQ[3];
    qQ[2] = lQ[4];

//	cout << Q << endl << endl;

    return Q;
}



cv::Mat RegionHandler::LS_Affine_Trans( std::vector<cv::Point2f> inPts, std::vector<cv::Point2f> outPts, std::vector<float> ptWeights )
{

    cv::Mat Q;
    cv::Mat A( cv::Size( 3 , inPts.size() ) , CV_32F);
    cv::Mat X( cv::Size( 2 , inPts.size() ) , CV_32F);

    for (int kk = 0; kk<ptOrig.size(); kk++)
    {
        float* Akk = A.ptr<float>(kk);
        float* Xkk = X.ptr<float>(kk);
        Akk[0] = ptOrig[kk].x * ptWeights[kk];
        Akk[1] = ptOrig[kk].y * ptWeights[kk];
        Akk[2] = ptWeights[kk];
        Xkk[0] = ptTrans[kk].x * ptWeights[kk];
        Xkk[1] = ptTrans[kk].y * ptWeights[kk];
    }

    Q = ((((A.t()*A).inv(cv::DECOMP_LU))*A.t())*X).t();

//	cout << "affine" << endl;
//	cout << Q << endl << endl;

    return Q;
}



Mat RegionHandler::LS_Rot_Trans( vector <Point2f> ptOrig, vector <Point2f> ptTrans, vector <float> ptWeights )
{
    // Q = [  1   b   e  ]
    //     [ -b   1   f  ]
    // A = [x  y  1  0  0]  ->  X = [ X ]
    //   = [0 -x  0  y  1]  ->      [ Y ]

    Mat linQ;
    Mat Q( Size(3,2) , CV_32F);
    Mat A( Size( 3 , 2*ptOrig.size() ) , CV_32F);
    Mat X( Size( 1 , 2*ptOrig.size() ) , CV_32F);

    for (int kk = 0; kk<ptOrig.size(); kk++)
    {
        float* Akk = A.ptr<float>(2*kk);
        float* Xkk = X.ptr<float>(2*kk);
        Akk[0] = ptOrig[kk].y * ptWeights[kk];
        Akk[1] = ptWeights[kk];
        Akk[2] = 0;
        Xkk[0] = (ptTrans[kk].x - ptOrig[kk].x )* ptWeights[kk];

        Akk = A.ptr<float>(2*kk+1);
        Xkk = X.ptr<float>(2*kk+1);
        Akk[0] = -ptOrig[kk].x * ptWeights[kk];
        Akk[1] = 0;
        Akk[2] = ptWeights[kk];
        Xkk[0] = (ptTrans[kk].y - ptOrig[kk].y )* ptWeights[kk];
    }
    linQ = ((((A.t()*A).inv(cv::DECOMP_LU))*A.t())*X).t();

    float* lQ = linQ.ptr<float>(0);
    float* qQ = Q.ptr<float>(0);
    qQ[0] = 1;
    qQ[1] = lQ[0];
    qQ[2] = lQ[1];
    qQ = Q.ptr<float>(1);
    qQ[0] = -lQ[0];
    qQ[1] = 1;
    qQ[2] = lQ[2];

//	cout << Q << endl << endl;

    return Q;
}

/*************************************************************************************************************************************************/

// align to the 'master region' if uniform timing is on
void RegionHandler::Reset_Timing( IlluminatedRegion& irMaster , int regionNo )
{
    if ( regionList.size() == 0 )  { return; }

    if (uniformTiming)
    {
        regionList[0].onTick = getTickCount()+100000;
        regionList[0].pulseOn = false;
        regionList[0].phaseShift = irMaster.phaseShift;
        regionList[0].pulseFreq = irMaster.pulseFreq;
        regionList[0].dutyCycle = irMaster.dutyCycle;

        for (int ii = 1; ii < regionList.size(); ii++)
        {
            regionList[ii].onTick = regionList[ii-1].onTick + irMaster.phaseShift*tickFreq/irMaster.pulseFreq;
            regionList[ii].pulseOn = false;
            regionList[ii].phaseShift = irMaster.phaseShift;
            regionList[ii].pulseFreq = irMaster.pulseFreq;
            regionList[ii].dutyCycle = irMaster.dutyCycle;
        }
    }
    else
    {
        if (regionNo < regionList.size() || regionNo == 0)
        {
            regionList[0].pulseOn = false;
            regionList[0].onTick = getTickCount()+10000;
            for (int ii = 1; ii < regionList.size(); ii++)
            {
                regionList[ii].onTick = regionList[ii-1].onTick + regionList[ii].phaseShift*tickFreq/regionList[ii-1].pulseFreq;
                regionList[ii].pulseOn = false;
            }
        }
        else if ( regionNo == regionList.size() )
        {
            regionList[regionNo].pulseOn = regionList[regionNo - 1].pulseOn;
            regionList[regionNo].offTick = regionList[regionNo-1].offTick + regionList[regionNo].phaseShift*tickFreq/regionList[regionNo-1].pulseFreq;
            regionList[regionNo].onTick = regionList[regionNo-1].onTick + regionList[regionNo].phaseShift*tickFreq/regionList[regionNo-1].pulseFreq;
        }
    }

    return;
}



void RegionHandler::Draw_Regions_On_Projector( Mat & projImage )
{
    int64 tickCount = getTickCount();
    for ( int ii=0 ; ii< regionList.size() ; ii++ )
    {
        regionList[ii].Update_Blink_State( tickCount );

        if ( regionList[ii].isActive )
        {
            if ( regionList[ii].pulseOn )
                regionList[ii].Draw_Transformed_Region( projImage, CV_FILLED);

            else if ( !blackWhenOffPrecedence && blackWhenOff )
                regionList[ii].Draw_Transformed_Region( projImage, CV_FILLED, Scalar(0,0,0) );
        }
    }

    // black precendence --> draw the black shapes last to overwrite any color
    if (blackWhenOffPrecedence && blackWhenOff)
        for ( int ii=0 ; ii<regionList.size() ; ii++ )
            if ( !regionList[ii].pulseOn && regionList[ii].isActive )
                regionList[ii].Draw_Transformed_Region( projImage, CV_FILLED, Scalar(0,0,0) );


}



void RegionHandler::Draw_Regions_On_Camera(cv::Mat& camImage, bool trackingOn)
{
    for (IlluminatedRegion region : regionList)
        if (region.isActive)  region.Draw_Region(camImage);

    if (showTrackPts && trackingOn)
        Draw_Track_Points(camImage);
}


void RegionHandler::Draw_Track_Points(cv::Mat& camImage)
{
    for (IlluminatedRegion region : regionList)
        if (region.isActive) region.Draw_Key_Points(camImage);
}
