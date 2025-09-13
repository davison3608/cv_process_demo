#include "ff_cv_packed.h"

namespace cv_cess {
//! 滤波操作
struct Filtercv {
    cv::Mat mean_;
    cv::Mat median_;
    cv::Mat gaussi_;
    
    void filter(cv::Mat *_mat);
};
//! 直方图操作
struct Histogramcv {
    cv::Mat histogram_; //像素分布表
    cv::Mat his_mean_; //对比增强灰度图

    void histogram(cv::Mat *_mat);
};
//! 阈值操作
struct Thresholdcv {
    cv::Mat _gray_binary; //二值化阈值
    cv::Mat _gary_binary_inv;
    cv::Mat _gary_trunc; //截断阈值
    cv::Mat _gary_tozero_inv;
    cv::Mat _gary_tozero;

    void thresh(cv::Mat *_mat);
};
//! 形态学操作
struct formcv {
    cv::Mat fushi_;
    cv::Mat pezhang_;
    cv::Mat bisuan_;
    cv::Mat kaisuan_;
    cv::Mat tidu_;
    cv::Mat dingmao_;
    cv::Mat heimao_;

    void form(cv::Mat *_mat);
};
//! 边缘检测
struct Edgecv {
    cv::Mat canny_;
    cv::Mat sobel_;
    cv::Mat scharr_;
    cv::Mat lap_;

    void edge(cv::Mat *_mat);
};
//! 轮廓检测与匹配
struct outlinecv {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat _gray_edge_mat; //灰度分割图
    cv::Mat _contours_mat; //标识轮廓的彩色图
    float _area;
    float _length;

    //传递原图 边缘处理后的灰度图
    void outline(cv::Mat *_mat, cv::Mat *_mat_cess);

    //传递另一个边缘灰度图
    double outline_compare_hu(cv::Mat *_ormat_cess);

    cv::Mat _contours_mat_match;
    cv::Point _best_loc;
    double _best_val;

    //输入模板边缘灰度图
    void outline_compare_matchshapes(cv::Mat *_templ_edge);
};
//! 特征检测与匹配
struct Featurecv {
    cv::Mat _harris;
    cv::Mat _shi_tomasi;
    cv::Mat _sift_gray;
    cv::Mat _surf_gray;
    cv::Mat _orb_gray;
    cv::Mat _fast_gray;
    cv::Mat _brief_gray;
    cv::Mat _akaze_gray;

    //输入原图像 灰度平滑直方图增强处理
    void feature(cv::Mat *_mat);

    cv::Mat _BFMatcher_pin;
    cv::Mat _FLAnn_pin;

    //输入原图像 默认orb检测后特征匹配
    void ture_compare(cv::Mat *_mat1, cv::Mat *_mat2);
};
//! 集合
class SimpCv
{
public:
    SimpCv() {}
    ~SimpCv() {}
    //! 原图像
    cv::Mat _rgb_img;
    cv::Mat _bgr_img;
    std::array<int, 2> _xysize{800, 600};
    //! 读取 精度与格式改变
    void read(const char *_path);

    //! 平滑操作
    Filtercv _Filtercv;
    //! 直方图增强
    Histogramcv _Histogramcv;
    //! 阈值操作
    Thresholdcv _Thresholdcv;
    //! 形态学操作
    formcv _formcv;
    //! 边缘检测
    Edgecv _Edgecv;
    //! 轮廓检测与匹配
    outlinecv _outlinecv;
    //! 特征检测与匹配
    Featurecv _Featurecv;
};
} // namespace cv_cess



//! 测试函数
void decode();
void transtream();



