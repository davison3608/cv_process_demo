#include "process.h"
int main(int argc, char const *argv[])
{
using namespace cv_cess;
using namespace std;
SimpCv *simcv=new SimpCv();
SimpCv *simcv2=new SimpCv();
simcv->read("../gile2.jpeg");
simcv2->read("../gile3.png");
simcv->_Filtercv.filter(&simcv->_rgb_img);
simcv->_Histogramcv.histogram(&simcv->_Filtercv.gaussi_);

cv::cvtColor(simcv->_Filtercv.gaussi_, simcv->_Filtercv.gaussi_, cv::COLOR_RGB2BGR);
cv::imshow("Display src 1", simcv->_bgr_img);
cv::imshow("Display src 2", simcv2->_bgr_img);

simcv->_Thresholdcv.thresh(&simcv->_Histogramcv.his_mean_);
simcv->_formcv.form(&simcv->_Thresholdcv._gary_trunc);
cv::imshow("闭运算", simcv->_formcv.bisuan_);

simcv->_Edgecv.edge(&simcv->_formcv.bisuan_);
cv::imshow("canny边缘", simcv->_Edgecv.canny_);
simcv->_outlinecv.outline(&simcv->_bgr_img, &simcv->_Edgecv.canny_);
cv::imshow("轮廓检测", simcv->_outlinecv._contours_mat);

//特征检测无需边缘处理
simcv->_Featurecv.feature(&simcv->_bgr_img);
cv::imshow("orb特征检测", simcv->_Featurecv._orb_gray);
simcv->_Featurecv.ture_compare(&simcv->_bgr_img, &simcv2->_bgr_img);
cv::imshow("bfmatch匹配", simcv->_Featurecv._BFMatcher_pin);
cv::imshow("flann匹配", simcv->_Featurecv._FLAnn_pin);

cv::waitKey(0);
delete simcv;
return 0;
}
