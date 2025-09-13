#include "process.h"

using std::string;
using std::cout;
using namespace cv_cess;

void SimpCv::read(const char *_path) {
    this->_rgb_img=cv::imread(_path);
    assert(!this->_rgb_img.empty());
    this->_rgb_img.convertTo(this->_rgb_img, CV_8U);
    cv::Size _size(this->_xysize[0], this->_xysize[1]);
    cv::resize(this->_rgb_img, this->_rgb_img, _size);
    cv::cvtColor(this->_rgb_img, this->_rgb_img, cv::COLOR_BGR2RGB);
    cv::cvtColor(this->_rgb_img, this->_bgr_img, cv::COLOR_RGB2BGR);
    return ;
}

void Thresholdcv::thresh(cv::Mat *_mat) {
    assert(_mat->channels() == 1);
    cv::Mat _gray;
    _gray=_mat->clone();
    //二值化阈值处理
    //高于阈值为maxval 低于为0 严格分割前后景 
    //文档二值化 物体检测
    cv::threshold(_gray, _gray_binary, 127, 255, cv::THRESH_BINARY); 
    //反二值化阈值处理
    //高于阈值为0 低于为maxval 
    //掩膜获取 暗部提取
 	cv::threshold(_gray, _gary_binary_inv, 127, 255, cv::THRESH_BINARY_INV); 
 	//截断阈值化处理
    //高于阈值截断thresh 低于为原值 抑制高光 保留暗部细节
    //高光抑制 明暗均衡
     cv::threshold(_gray, _gary_trunc, 127, 255, cv::THRESH_TRUNC); 
    //超阈值零处理
    //高于阈值为0 低于为原值 消除亮区噪声 保留暗部细节
    //过曝修复 亮部视为噪声去除
 	cv::threshold(_gray, _gary_tozero_inv, 127, 255, cv::THRESH_TOZERO_INV); 
    //低阈值零处理
    //高于阈值保留原值 低于设为0 提取亮部 忽略暗部
    //灯源部分检测 暗部视为噪声去除
 	cv::threshold(_gray, _gary_tozero, 127, 255, cv::THRESH_TOZERO); 
}

void Filtercv::filter(cv::Mat *_mat) {
    //均值滤波 
    //轻度噪声平滑 计算快速 边缘模糊明显
    cv::blur(*_mat, this->mean_, cv::Size(3, 3));
    //中值滤波
    //椒盐或是脉冲类孤立噪声去除 对大噪声效果不好
    cv::medianBlur(*_mat, this->median_, 3);
    //高斯滤波
    //自然噪声去除 边缘保留平滑 计算量更大
    cv::GaussianBlur(*_mat, this->gaussi_, cv::Size(3, 3), 1.5);
}

void Histogramcv::histogram(cv::Mat *_mat) {
    cv::Mat _gray;
    cv::cvtColor(*_mat, _gray, cv::COLOR_BGR2GRAY);
    //计算直方图
    //图像像素强度分布的图形表示 横轴表示像素强度值 纵轴表示该强度值的像素数量
    int histSize = 256; //直方图的bin数量（0-255）
    float range[] = {0, 256}; //像素值范围
    const float* histRange = {range};
    cv::Mat hist;
    //输入图像 图像数量 通道索引 掩码 输出直方图 维度 bin数量 范围
    cv::calcHist(&_gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    //归一化直方图 将直方图高度缩放到绘图区域内
    int histHeight = 400; //直方图图像的高度
    int histWidth = 512; //直方图图像的宽度
    cv::normalize(hist, hist, 0, histHeight, cv::NORM_MINMAX, -1, cv::Mat());
    //创建直方图图像 黑色背景
    cv::Mat histMat(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    //绘制直方图的每个bin 柱状图
    int binWidth = cvRound((double)histWidth / histSize); // 每个bin的宽度
    for (int i = 1; i < histSize; i++) {
    //绘制当前bin与前一个bin之间的线段 形成柱形
    cv::line(
        histMat,
        cv::Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
        cv::Point(binWidth * i, histHeight - cvRound(hist.at<float>(i))),
        cv::Scalar(0, 255, 0), // 绿色线条
        2
    ); //线条宽度
    }
    this->histogram_=histMat.clone();
    //直方图均衡化
    //重新分配像素强度值 使直方图更加均匀 增强明暗对比度
    //图像增强 通过直方图均衡化 可以增强图像的对比度 使细节更加清晰
    //图像分割 过分析直方图 可以确定阈值 用于图像分割
    //图像匹配 通过比较直方图 可以判断两幅图像的相似度 用于图像匹配和检索
    //颜色分析 通过颜色直方图 可以分析图像的颜色分布 用于颜色校正和风格化处理
    cv::equalizeHist(_gray, this->his_mean_);
}

void formcv::form(cv::Mat *_mat) {
    assert(_mat->channels() == 1);
    cv::Mat _gray;
    _gray=_mat->clone();
    //卷积核定义
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //腐蚀
    //图像卷积 当核覆盖前景像素时中心像素保留 否则腐蚀掉
    //去除噪声 分离不同物体
    cv::erode(_gray, fushi_, kernel);
    //膨胀
    //图像卷积 当核与前景像素重叠时中心像素就保留
    //连接断裂的物体 补全空洞
    cv::dilate(_gray, pezhang_, kernel);
    //闭运算
    //先腐蚀后膨胀
    //去除小物体并平滑边界
    cv::morphologyEx(_gray, bisuan_, cv::MORPH_CLOSE, kernel);
    //开运算
    //先膨胀后腐蚀
    //填充空洞连接物体
    cv::morphologyEx(_gray, bisuan_, cv::MORPH_OPEN, kernel);
    //形态梯度
    //膨胀图减去腐蚀图 提取物体边缘 突出轮廓
    cv::morphologyEx(_gray, tidu_, cv::MORPH_GRADIENT, kernel);
    //顶帽运算
    //原图减去开运算图 提取比背景量亮的细小物体 增强细小亮部
    cv::morphologyEx(_gray, dingmao_, cv::MORPH_TOPHAT, kernel);
    //黑帽运算
    //闭运算图减去原图 提取比背景暗的细小物体 增强细小暗部
    cv::morphologyEx(_gray, heimao_, cv::MORPH_BLACKHAT, kernel);
}

void Edgecv::edge(cv::Mat *_mat) {
    assert(_mat->channels() == 1);
    cv::Mat _gray;
    _gray=_mat->clone();
    //canny方法
    //多阶段运算 检测效果好 抑制噪声效果好 通用的边缘检测方法
    cv::Canny(_gray, canny_, 50, 150, 3);
    //sobel方法
    //基于一阶导数运算 适合垂直于水平方向的边缘提取
    cv::Sobel(_gray, sobel_, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(sobel_, sobel_); //转换为8位图像
    //scharr方法
    //对sobel的改进 能够提取细小边缘
    cv::Scharr(_gray, scharr_, CV_16S, 1, 0);
    cv::convertScaleAbs(scharr_, scharr_);
    //Laplacian方法
    //提取边缘与脚点 基于二阶导数 但是对噪声敏感
    cv::Laplacian(_gray, lap_, CV_16S, 3, 1, 0);
    cv::convertScaleAbs(lap_, lap_);
}

void outlinecv::outline(cv::Mat *_mat, cv::Mat *_mat_cess) {
    this->_gray_edge_mat=_mat_cess->clone();
    //查找图像轮廓 存储轮廓与层级
    cv::findContours(
        *_mat_cess, this->contours, this->hierarchy, 
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE
    );
    //绘制图像轮廓
    this->_contours_mat=_mat->clone();
    cv::drawContours(this->_contours_mat, this->contours, -1, cv::Scalar(0, 0, 255), 2);
    //轮廓属性 面积与周长计算
    if (!this->contours.empty()) {
    //找到面积最大的轮廓
    int max_idx = 0;
    double max_area = 0;
    for (size_t i = 0; i<this->contours.size(); ++i) {
    double area = cv::contourArea(this->contours[i]);
    if (area > max_area) {
        max_area = area;
        max_idx = i;
    }
    }
    // 计算最大轮廓的面积和周长
    this->_area = static_cast<float>(max_area);
    // 第二个参数为true表示轮廓是闭合的
    this->_length = static_cast<float>(cv::arcLength(this->contours[max_idx], true));
    } 
    else {
    // 没有找到轮廓时初始化
    this->_area = 0;
    this->_length = 0;
    }
}

double outlinecv::outline_compare_hu(cv::Mat *_ormat_cess) {
    //hu矩匹配
    //基于形状的全局统计特征 对缩放或者旋转的容忍度更高
    //忽略局部细节关注整体 适于图像分类 通用目标识别
    if (_ormat_cess->empty())
    return INFINITY;
    std::vector<std::vector<cv::Point>> orcontours;
    std::vector<cv::Vec4i> orhierarchy;
    cv::findContours(
        *_ormat_cess, orcontours, orhierarchy, 
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE
    );
    //找到二者最大轮廓
    int max_idx = 0;
    int ormax_idx = 0;
    if (!this->contours.empty()) {
    double max_area = 0;
    for (size_t i = 0; i<this->contours.size(); ++i) {
    double area = cv::contourArea(this->contours[i]);
    if (area > max_area) {
        max_area = area;
        max_idx = i;
        }
    }
    }
    if (!orcontours.empty()) {
    double max_area = 0;
    for (size_t i = 0; i<orcontours.size(); ++i) {
    double area = cv::contourArea(orcontours[i]);
    if (area > max_area) {
        max_area = area;
        ormax_idx = i;
        }
    }
    }
    auto&contour1=this->contours[max_idx];
    auto&contour2=orcontours[ormax_idx];
    //计算两个轮廓的矩和Hu矩
    cv::Moments moments1 = cv::moments(contour1);
    cv::Moments moments2 = cv::moments(contour2);
    double hu1[7], hu2[7];
    cv::HuMoments(moments1, hu1);
    cv::HuMoments(moments2, hu2);
    //对Hu矩取对数 增强区分度 处理数值过小的问题
    for (int i = 0; i < 7; ++i) {
    //取负号使数值为正 便于比较
    //加小值避免log(0)
    hu1[i]=-std::copysign(1.0, hu1[i]) * std::log10(std::abs(hu1[i]) + 1e-10); 
    hu2[i]=-std::copysign(1.0, hu2[i]) * std::log10(std::abs(hu2[i]) + 1e-10);
    }
    //将double数组转换为cv::Mat 再计算距离
    cv::Mat hu_mat1(7, 1, CV_64F, hu1);  
    cv::Mat hu_mat2(7, 1, CV_64F, hu2);
    //计算欧氏距离（L2范数）作为相似度度量
    //距离越小表示越相似
    return cv::norm(hu_mat1, hu_mat2, cv::NORM_L2);
}

void outlinecv::outline_compare_matchshapes(cv::Mat *_templ_edge) {
    //模版匹配
    //基于轮廓直接距离的度量 对极端缩放旋转的容忍度低
    //对局部细节更敏感 适用于特定目标定位
    cv::Mat result;  
    int result_cols=this->_gray_edge_mat.cols - _templ_edge->cols + 1;
    int result_rows=this->_gray_edge_mat.rows - _templ_edge->rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);  // 热力图数据类型

    //选择匹配方法
    int match_method=cv::TM_CCOEFF_NORMED;
    cv::matchTemplate(this->_gray_edge_mat, *_templ_edge, result, match_method);

    //从热力图中找最佳匹配位置
    double min_val, max_val;  //存储最小/最大相似度值
    cv::Point min_loc, max_loc;   //存储最小/最大相似度对应的位置
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    //确定最佳匹配位置（不同方法的判断逻辑不同）
    if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED) {
    this->_best_loc = min_loc;  //值越小越相似
    this->_best_val=min_val;
    } 
    else {
    this->_best_loc = max_loc;  //其他方法是值越大越相似
    this->_best_val=max_val;
    }
    cout<<"最佳匹配位置坐标 "<<this->_best_loc.x<<this->_best_loc.y<<"\n";
    cout<<"最佳匹配的相似度值 "<<this->_best_val<<'\n';
}

void Featurecv::feature(cv::Mat *_mat) {
    cv::Mat _gray;
    cv::cvtColor(*_mat, _gray, cv::COLOR_BGR2GRAY);
    //高斯滤波 直方图增强
    cv::GaussianBlur(_gray, _gray, cv::Size(3, 3), 1.5);
    cv::equalizeHist(_gray, _gray);
    
    //无描述子仅有角点 无法用于匹配
    //Harris角点检测
    //刚性物体运动估计 低分辨率光照固定场景 对缩放尺度和噪声敏感
    cv::cornerHarris(_gray, this->_harris, 2, 3, 0.04);
    //归一化结果 像素值控制在0~255
    this->_harris=_gray.clone();
    cv::normalize(this->_harris, this->_harris, 0, 255, cv::NORM_MINMAX);
    
    //shi Tomasi角点检测
    //harris改进 计算稍大 实时性差
    std::vector<cv::Point> corners;
    cv::goodFeaturesToTrack(_gray, corners, 100, 0.1, 10);
    //绘制角点
    this->_shi_tomasi=_gray.clone();
    for (auto&p: corners) 
    cv::circle(this->_shi_tomasi, p, 5, cv::Scalar(0, 0, 255), 2);
    
    //高精度有限 噪声敏感 检测不稳定
    {
    //sfit算法
    //高精度图像匹配 复杂光照/尺度变化下的目标识别 计算量大实时性差 对模糊图像容忍度差 

    //surf算法
    //改进sfit 计算速度更快 对噪声更为敏感
    }

    //精度适中 实时性优先
    {
    //orb算法
    //实时性好 支持尺度、旋转、光照不变性 但是对于低纹理图像匹配点较少
    std::vector<cv::KeyPoint> orb_keypoints;
    cv::Mat orb_descriptors;
    cv::Ptr<cv::ORB> orb=cv::ORB::create();
    this->_orb_gray=_gray.clone();
    orb->detectAndCompute(this->_orb_gray, cv::noArray(), orb_keypoints, orb_descriptors);
    //绘制角点
    cv::drawKeypoints(this->_orb_gray, orb_keypoints, this->_orb_gray);

    //fast算法
    //计算速度极快 但是无描述子 需要orb或者brief描述子 检测角点分布不均 容易高纹理区域聚集
    
    //brief算法
    //实时性好 内存占用低 但是无旋转不变性 旋转后匹配率骤降 并对光照对比度敏感
    }

    //akaze算法
    //实时性一般 比sift更好地支持尺度、旋转、光照、仿射不变性 复杂形变场景 对模糊图像容忍度更差于sift
    std::vector<cv::KeyPoint> ak_keypoints;
    cv::Mat ak_descriptors;
    cv::Ptr<cv::AKAZE> akaze=cv::AKAZE::create();
    this->_akaze_gray=_gray.clone();
    akaze->detectAndCompute(this->_akaze_gray, cv::noArray(), ak_keypoints, ak_descriptors);
    //绘制角点
    cv::drawKeypoints(this->_akaze_gray, ak_keypoints, this->_akaze_gray);
}

void Featurecv::ture_compare(cv::Mat *_mat1, cv::Mat *_mat2)
{
    assert(!_mat1->empty() && !_mat2->empty());
    cv::Mat _gray1, _gray2;
    cv::cvtColor(*_mat1, _gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(*_mat2, _gray2, cv::COLOR_BGR2GRAY);
    //高斯滤波 直方图增强
    cv::GaussianBlur(_gray1, _gray1, cv::Size(3, 3), 1.5);
    cv::GaussianBlur(_gray2, _gray2, cv::Size(3, 3), 1.5);
    cv::equalizeHist(_gray1, _gray1);
    cv::equalizeHist(_gray2, _gray2);
    //akaze角点检测
    std::vector<cv::KeyPoint> ak_keypoints1;
    std::vector<cv::KeyPoint> ak_keypoints2;
    cv::Mat ak_descriptors1;
    cv::Mat ak_descriptors2;
    cv::Ptr<cv::AKAZE> ak1=cv::AKAZE::create(    
        cv::AKAZE::DESCRIPTOR_KAZE,  //使用浮点型描述子
        0, 3, 0.001f
    );
    cv::Ptr<cv::AKAZE> ak2=cv::AKAZE::create(
        cv::AKAZE::DESCRIPTOR_KAZE,  //使用浮点型描述子
        0, 3, 0.001f
    );
    ak1->detectAndCompute(_gray1, cv::noArray(), ak_keypoints1, ak_descriptors1);
    ak2->detectAndCompute(_gray2, cv::noArray(), ak_keypoints2, ak_descriptors2);
    //绘制带有角点图
    cv::drawKeypoints(_gray1, ak_keypoints1, _gray1);
    cv::drawKeypoints(_gray2, ak_keypoints2, _gray2);

    //BFMatcher匹配
    //暴力遍历所有特征点 计算所有描述子距离 保留最相似匹配
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(ak_descriptors1, ak_descriptors2, matches);
    cv::drawMatches(
        *_mat1, ak_keypoints1,  // 第一张原图及其特征点
        *_mat2, ak_keypoints2,  // 第二张原图及其特征点
        matches,                 // 匹配结果
        this->_BFMatcher_pin,    // 输出图像
        cv::Scalar(0, 255, 0),   // 匹配线颜色（绿色）
        cv::Scalar(0, 0, 255),   // 未匹配点颜色（红色）
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  // 不绘制未匹配的点
    );

    //flann匹配
    //基于近似最近邻算法kd-tree、LSH等 通过构建索引加速搜索 牺牲部分精度换取速度
    cv::FlannBasedMatcher matcher_fl;
    std::vector<cv::DMatch> matches_fl;
    matcher_fl.match(ak_descriptors1, ak_descriptors2, matches_fl);
    cv::drawMatches(
        *_mat1, ak_keypoints1,  // 第一张原图及其特征点
        *_mat2, ak_keypoints2,  // 第二张原图及其特征点
        matches_fl,              // 匹配结果
        this->_FLAnn_pin,    // 输出图像
        cv::Scalar(0, 255, 0),   // 匹配线颜色（绿色）
        cv::Scalar(0, 0, 255),   // 未匹配点颜色（红色）
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  // 不绘制未匹配的点
    );
}

