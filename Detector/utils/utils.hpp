
#pragma once
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <iomanip> // std::fixed, std::setprecision
#include <sstream> // std::ostringstream
#include <map>
#include <chrono>
#include <cmath>
#include <algorithm>

std::vector<std::vector<int>> COLOR_TABLE = {
    {0,   114, 189},    {217,  83,  25},    {237, 176,  32},    {126,  47, 142},    {119, 172,  48},    { 77, 190, 238},
    {162,  20,  47},    { 77,  77,  77},    {153, 153, 153},    {255,   0,   0},    {255, 128,   0},    {191, 191,   0},
    {  0, 255,   0},    {  0,   0, 255},    {170,   0, 255},    { 85,  85,   0},    { 85, 170,   0},    { 85, 255,   0},
    {170,  85,   0},    {170, 170,   0},    {170, 255,   0},    {255,  85,   0},    {255, 170,   0},    {255, 255,   0},
    {  0,  85, 128},    {  0, 170, 128},    {  0, 255, 128},    { 85,   0, 128},    { 85,  85, 128},    { 85, 170, 128},
    { 85, 255, 128},    {170,   0, 128},    {170,  85, 128},    {170, 170, 128},    {170, 255, 128},    {255,   0, 128},
    {255,  85, 128},    {255, 170, 128},    {255, 255, 128},    {  0,  85, 255},    {  0, 170, 255},    {  0, 255, 255},
    { 85,   0, 255},    { 85,  85, 255},    { 85, 170, 255},    { 85, 255, 255},    {170,   0, 255},    {170,  85, 255},
    {170, 170, 255},    {170, 255, 255},    {255,   0, 255},    {255,  85, 255},    {255, 170, 255},    { 85,   0,   0},
    {128,   0,   0},    {170,   0,   0},    {213,   0,   0},    {255,   0,   0},    {  0,  43,   0},    {  0,  85,   0},
    {  0, 128,   0},    {  0, 170,   0},    {  0, 213,   0},    {  0, 255,   0},    {  0,   0,  43},    {  0,   0,  85},
    {  0,   0, 128},    {  0,   0, 170},    {  0,   0, 213},    {  0,   0, 255},    {  0,   0,   0},    { 36,  36,  36},
    { 73,  73,  73},    {109, 109, 109},    {146, 146, 146},    {182, 182, 182},    {219, 219, 219},    {  0, 114, 189},
    { 80, 183, 189},    {128, 128,   0} // (0.50, 0.5, 0) → (128,128,0)
};

const std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

const std::map<int, std::string> COCO_LABELS_RFDETR = {
    {1,"person"}, {2,"bicycle"}, {3,"car"}, {4,"motorcycle"}, {5,"airplane"}, {6,"bus"}, {7,"train"}, {8,"truck"},
    {9,"boat"}, {10, "traffic light"}, {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
    {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"}, {22, "elephant"}, {23, "bear"},
    {24, "zebra"}, {25, "giraffe"}, {27, "backpack"}, {28, "umbrella"}, {31, "handbag"}, {32, "tie"},{33, "suitcase"}, 
    {34, "frisbee"}, {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"}, {40, "baseball glove"}, 
    {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"}, {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"}, 
    {49, "knife"}, {50, "spoon"}, {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"}, {56, "broccoli"}, 
    {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"}, {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, 
    {65, "bed"}, {67, "dining table"}, {70, "toilet"}, {72, "tv"}, {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"}, 
    {77, "cell phone"}, {78, "microwave"}, {79, "oven"}, {80, "toaster"}, {81, "sink"}, {82, "refrigerator"}, {84, "book"}, {85, "clock"}, 
    {86, "vase"}, {87, "scissors"}, {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"},
};

std::vector<std::string>  VisDrone_LABELS = {
    "regions", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", 
    "bus", "motor", "others"
};

std::vector<std::string>  AITOD_LABELS = {
    "airplane", "bridge", "storage tank", "ship", "swimming pool", "vehicle", "person", "wind mill"
};

void gen_dir(std::string engine_dir_path)
{
    if (mkdir(engine_dir_path.c_str(), 0777) == 0)
    {
        std::cout << "generated directory :: " << engine_dir_path << std::endl;
    }
    else
    {
        std::cerr << "already exist" << std::endl;
    }
}

bool isImageFile(const std::string &filename)
{
    const std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};

    for (const auto &ext : imageExtensions)
    {
        if (filename.size() >= ext.size() &&
            filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0)
        {
            return true;
        }
    }
    return false;
}

void load_images_from_folder(std::vector<std::string> &file_names, const std::string &folderPath)
{
    // Open directory pointer
    DIR *dir = opendir(folderPath.c_str());
    if (dir == nullptr)
    {
        std::cerr << "[ERROR] Cannot open directory: " << folderPath << std::endl;
        exit(EXIT_FAILURE);
    }
    // Read directory entries
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (isImageFile(filename))
        {
            // std::cout << filename << std::endl;
            file_names.push_back(filename);
        }
    }
    // Close the directory
    closedir(dir);
}

void show_image(cv::Mat img, std::string window_name, int waitkey=1, float scale=1.0) 
{
    cv::resize(img, img, cv::Size(static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale)));
    cv::namedWindow(window_name);
    cv::moveWindow(window_name, 30, 30);
    cv::imshow(window_name, img);
    cv::waitKey(waitkey);
    cv::destroyAllWindows();
}

// preprocess for yolo11, yolo12
/*
    INPUT  = BGR[NHWC](0, 255)
    OUTPUT = RGB[NCHW](0.f,1.f)
    This equation include 5 steps
    0. resize
    1. letterbox padding
    2. Scale Image to range [0.f, 1.0f], /255.f
    3. Shuffle form HWC to CHW
    4. BGR -> RGB
*/
void pre_proc_yolo(    
    std::vector<float> &output,     // output float data
    cv::Mat &ori_img,               // input image
    float &ratio,                   // ratio 
    std::vector<float> &pad_tops,   // pad_tops 
    std::vector<float> &pad_lefts,  // pad_lefts 
    int b_idx, int INPUT_SIZE, int INPUT_H, int INPUT_W)
{
    int image_h = ori_img.rows;
    int image_w = ori_img.cols;

    int resized_image_h = static_cast<int>(std::round(image_h * ratio));
    int resized_image_w = static_cast<int>(std::round(image_w * ratio));
    
    // resize
    cv::Mat resized_img;
    if ((int)INPUT_W != resized_image_w || (int)INPUT_H != resized_image_h){
        cv::resize(ori_img, resized_img, cv::Size(resized_image_w, resized_image_h));
    }
    else{
        resized_img = ori_img.clone();
    }

    // letterbox padding
    int pad_w = INPUT_W - resized_image_w;
    int pad_h = INPUT_H - resized_image_h;
    int pad_left   = static_cast<int>(pad_w / 2);
    int pad_right  = static_cast<int>(pad_w - pad_left);
    int pad_top    = static_cast<int>(pad_h / 2);
    int pad_bottom = static_cast<int>(pad_h - pad_top);
    cv::copyMakeBorder(resized_img, resized_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, { 114, 114, 114 });
    pad_tops.push_back(pad_top);
    pad_lefts.push_back(pad_left);

    // scale=1/255, BGR->RGB, HWC->CHW, float32
    cv::Mat out;
    cv::dnn::blobFromImage(resized_img, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    memcpy(output.data() + b_idx * INPUT_SIZE, out.data, INPUT_SIZE * sizeof(float));
};

// preprocess for yolox
/*
    INPUT  = BGR[NHWC](0, 255)
    OUTPUT = BGR[NCHW](0, 255)
    This equation include 3 steps
    0. resize
    1. padding (right, bottom) with (114, 114, 114)
    2. Shuffle form HWC to CHW
*/
void pre_proc_yolox(    
    std::vector<float> &output,     // output float data
    cv::Mat &ori_img,               // input image
    float &ratio,                   // ratio
    int b_idx, int INPUT_SIZE, int INPUT_H, int INPUT_W)
{
    int unpad_w = ratio * ori_img.cols;
    int unpad_h = ratio * ori_img.rows;

    // resize
    cv::Mat resized_img(unpad_h, unpad_w, CV_8UC3);
    if ((int)ori_img.cols != unpad_w || (int)ori_img.rows != unpad_h){
        cv::resize(ori_img, resized_img, resized_img.size());
    }
    else{
        resized_img = ori_img.clone();
    }

    // pad
    int pad_right = INPUT_W - (int)resized_img.cols;
    int pad_bottom = INPUT_H - (int)resized_img.rows;
    cv::copyMakeBorder(resized_img, resized_img, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT, { 114, 114, 114 });

    // HWC->CHW, float32
    cv::Mat out;
    cv::dnn::blobFromImage(resized_img, out, 1.f, cv::Size(), cv::Scalar(0, 0, 0), false, false, CV_32F);

    memcpy(output.data() + b_idx * INPUT_SIZE, out.data, INPUT_SIZE * sizeof(float));
};


// preprocess for yolox bytetrack
/*
    INPUT  = BGR[NHWC](0, 255)
    OUTPUT = RGB[NCHW](0.f,1.f)
    This equation include 6 steps
    1. resize
    2. padding (right, bottom) with (114, 114, 114)
    3. convert to float32 and scale 0~1
    4. BGR -> RGB
    5. Shuffle form HWC to CHW
    6. Normalize: (x - mean) / std
*/
void pre_proc_yolox_bt(    
    std::vector<float> &output,     // output float data
    cv::Mat &ori_img,               // input image
    float &ratio,                   // ratio
    int b_idx, int INPUT_SIZE, int INPUT_H, int INPUT_W,
    const std::vector<float>& mean,  // mean values (size 3)
    const std::vector<float>& std    // std values (size 3)
){
    // 1. resize with ratio
    int unpad_w = static_cast<int>(ratio * ori_img.cols);
    int unpad_h = static_cast<int>(ratio * ori_img.rows);

    cv::Mat resized_img;
    if (ori_img.cols != unpad_w || ori_img.rows != unpad_h) {
        cv::resize(ori_img, resized_img, cv::Size(unpad_w, unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = ori_img;
    }

    // 2. pad with 114
    int pad_right  = INPUT_W - resized_img.cols;
    int pad_bottom = INPUT_H - resized_img.rows;
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, 0, pad_bottom, 0, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 3. convert to float32 and scale 0~1
    cv::Mat float_img;
    padded_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 4. BGR -> RGB
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    // 5. HWC -> CHW directly into output
    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; c++) {
        chw[c] = cv::Mat(INPUT_H, INPUT_W, CV_32F,
                         output.data() + b_idx * INPUT_SIZE + c * INPUT_H * INPUT_W);
    }
    cv::split(float_img, chw);

    // 6. Normalize: (x - mean) / std
    for (int c = 0; c < 3; c++) {
        float* ptr = output.data() + b_idx * INPUT_SIZE + c * INPUT_H * INPUT_W;
        int channel_size = INPUT_H * INPUT_W;
        for (int i = 0; i < channel_size; i++) {
            ptr[i] = (ptr[i] - mean[c]) / std[c];
        }
    }
}

// preprocess for dfine, deim, rt_detr
/*
    INPUT  = BGR[NHWC](0, 255)
    OUTPUT = RGB[NCHW](0.f,1.f)
    This equation include 4 steps
    0. resize
    1. Scale Image to range [0.f, 1.0f], /255.f
    2. Shuffle form HWC to CHW
    3. BGR -> RGB
*/
void pre_proc_detr0(    
    std::vector<float> &output,     // output float data
    cv::Mat &ori_img,               // input image
    int b_idx, int INPUT_SIZE, int INPUT_H, int INPUT_W)
{
    int image_h = ori_img.rows;
    int image_w = ori_img.cols;

    // resize
    cv::Mat resized_img;
    if ((int)INPUT_W != image_w || (int)INPUT_H != image_h){
        cv::resize(ori_img, resized_img, cv::Size(INPUT_W, INPUT_H));
    }
    else{
        resized_img = ori_img.clone();
    }

    // scale=1/255, BGR->RGB, HWC->CHW, float32
    cv::Mat out;
    cv::dnn::blobFromImage(resized_img, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    memcpy(output.data() + b_idx * (INPUT_SIZE + 2 + 2), out.data, INPUT_SIZE);
};


void pre_proc_detr(    
    std::vector<float> &output,     // output float data
    const cv::Mat &ori_img,         // input image
    int b_idx,                      // batch index
    int INPUT_SIZE,                 // 3*H*W
    int INPUT_H, int INPUT_W        // target size
) {
    // 1. resize
    cv::Mat resized_img;
    if (INPUT_W != ori_img.cols || INPUT_H != ori_img.rows) {
        cv::resize(ori_img, resized_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);
    } else {
        resized_img = ori_img.clone();
    }

    // 2. float 변환 & [0,1] scaling
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // 3. BGR -> RGB
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    // 4. HWC -> CHW, output에 채우기
    for (int c = 0; c < 3; c++) {
        float* dst_ptr = output.data() + b_idx * INPUT_SIZE + c * INPUT_H * INPUT_W;
        for (int h = 0; h < INPUT_H; h++) {
            const float* src_ptr = float_img.ptr<float>(h);
            for (int w = 0; w < INPUT_W; w++) {
                dst_ptr[h * INPUT_W + w] = src_ptr[w * 3 + c];
            }
        }
    }
}

// preprocess for rf_detr
/*
    INPUT  = BGR[NHWC](0, 255)
    OUTPUT = RGB[NCHW](0.f,1.f)
    This equation include 4 steps
    0. resize 
    1. Scale Image to range [0.f, 1.0f], /255.f
    2. Shuffle form HWC to CHW
    3. BGR -> RGB
    4. Normalize: (x - mean) / std
*/
void pre_proc_rf_detr(
    std::vector<float> &output,      // output float buffer
    const cv::Mat &ori_img,          // input image
    int b_idx,                       // batch index
    int INPUT_SIZE,                  // total float size per image (3*H*W)
    int INPUT_H, int INPUT_W,        // resize target
    const std::vector<float>& mean,  // mean values (size 3)
    const std::vector<float>& std    // std values (size 3)
) {
    // 1. Resize
    cv::Mat resized_img;
    cv::resize(ori_img, resized_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);

    // 2. Convert to float and scale 0~1
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // scale

    // 2. Convert BGR->RGB
    cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

    // 3. HWC -> CHW
    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; i++) {
        chw[i] = cv::Mat(INPUT_H, INPUT_W, CV_32F, output.data() + b_idx * INPUT_SIZE + i * INPUT_H * INPUT_W);
    }
    cv::split(float_img, chw);

    // 4. Normalize: (x - mean) / std
    for (int c = 0; c < 3; c++) {
        float* ptr = output.data() + b_idx * INPUT_SIZE + c * INPUT_H * INPUT_W;
        int channel_size = INPUT_H * INPUT_W;
        for (int i = 0; i < channel_size; i++) {
            ptr[i] = (ptr[i] - mean[c]) / std[c];
        }
    }
}

void draw_bbox_text_yolo(
    float* detection_ptr,
    int num_dets,
    cv::Mat &img, 
    float ratio, int pad_top, int pad_left,
    float conf_thre, const std::string &img_name, const std::string &save_dir_path, 
    const std::vector<std::vector<int>> &color_table, const std::vector<std::string> &class_names)
{
    int x, y, x1, y1, w, h, cls_id;
    float conf;
    for (int d_idx = 0; d_idx < num_dets; d_idx++)
    {   
        int g_idx = d_idx * 6;
        conf = detection_ptr[g_idx + 4];
        if (conf < conf_thre) continue;
        x = static_cast<int>((detection_ptr[g_idx] - pad_left) / ratio);
        y = static_cast<int>((detection_ptr[g_idx + 1] - pad_top) / ratio) ;
        x1 = static_cast<int>((detection_ptr[g_idx + 2] - pad_left) / ratio);
        y1 = static_cast<int>((detection_ptr[g_idx + 3] - pad_top) / ratio);
        cls_id = static_cast<int>(detection_ptr[g_idx + 5]);
        w = x1 - x;
        h = y1 - y;

        // bbox
        cv::Rect rect(x, y, w, h);
        auto color_type = color_table[cls_id % color_table.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string str_conf = oss.str();
        std::string text = std::to_string(d_idx) + " " + class_names[cls_id] + " " + str_conf;
        cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1, 0);

        // print to console 
        std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
        std::to_string(cls_id) + ", " + std::to_string(conf) + ", " + class_names[cls_id];
        std::cout << consol_text << std::endl;
    }
}

void draw_bbox_text_yolox(
    float* detection_ptr,
    int num_dets,
    cv::Mat &img, 
    float ratio, float conf_thre, const std::string &img_name, const std::string &save_dir_path, 
    const std::vector<std::vector<int>> &color_table, const std::vector<std::string> &class_names, bool consol_print=true, int* det_count = nullptr)
{
    int x, y, x1, y1, w, h, cls_id;
    float conf;
    for (int d_idx = 0; d_idx < num_dets; d_idx++)
    {   
        int g_idx = d_idx * 6;
        conf = detection_ptr[g_idx + 4];
        if (conf < conf_thre) continue;
        x = static_cast<int>(detection_ptr[g_idx] / ratio);
        y = static_cast<int>(detection_ptr[g_idx + 1] / ratio);
        x1 = static_cast<int>(detection_ptr[g_idx + 2] / ratio);
        y1 = static_cast<int>(detection_ptr[g_idx + 3] / ratio);
        cls_id = static_cast<int>(detection_ptr[g_idx + 5]);
        w = x1 - x;
        h = y1 - y;

        // bbox
        cv::Rect rect(x, y, w, h);
        auto color_type = color_table[cls_id % color_table.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string str_conf = oss.str();
        std::string text = std::to_string(d_idx) + " " + class_names[cls_id] + " " + str_conf;
        cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1, 0);

        // print to console 
        if (consol_print){
            std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
            std::to_string(cls_id) + ", " + std::to_string(conf) + ", " + class_names[cls_id];
            std::cout << consol_text << std::endl;
        }
        if(det_count != nullptr)
            (*det_count)++;
    }
}

void draw_bbox_text_detr(
    float* detection_ptr,
    cv::Mat &img, 
    float conf_thre, const std::string &img_name, const std::string &save_dir_path, 
    const std::vector<std::vector<int>> &color_table, const std::vector<std::string> &class_names, bool consol_print=true, int* det_count = nullptr)
{
    int x, y, x1, y1, w, h, cls_id;
    float conf;
    for (int d_idx = 0; d_idx < 300; d_idx++)
    {   
        int g_idx = d_idx * 6;
        conf = detection_ptr[g_idx + 4];
        if (conf < conf_thre) continue;
        x = static_cast<int>(detection_ptr[g_idx]);
        y = static_cast<int>(detection_ptr[g_idx + 1]);
        x1 = static_cast<int>(detection_ptr[g_idx + 2]);
        y1 = static_cast<int>(detection_ptr[g_idx + 3]);
        cls_id = static_cast<int>(detection_ptr[g_idx + 5]);
        w = x1 - x;
        h = y1 - y;

        // bbox
        cv::Rect rect(x, y, w, h);
        auto color_type = color_table[cls_id % color_table.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string str_conf = oss.str();
        std::string text = std::to_string(d_idx) + " " + class_names[cls_id] + " " + str_conf;
        cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1, 0);

        // print to console 
        if (consol_print){
            std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
            std::to_string(cls_id) + ", " + std::to_string(conf) + ", " + class_names[cls_id];
            std::cout << consol_text << std::endl;
        }

        if(det_count != nullptr)
            (*det_count)++;
    }
}

void draw_bbox_text_rf_detr(
    float* detection_ptr,
    cv::Mat &img, 
    float conf_thre, const std::string &img_name, const std::string &save_dir_path, 
    const std::vector<std::vector<int>> &color_table, const std::map<int, std::string> &class_names)
{
    int x, y, x1, y1, w, h, cls_id, index;
    float conf;
    for (int d_idx = 0; d_idx < 300; d_idx++)
    {   
        int g_idx = d_idx * 6;
        conf = detection_ptr[g_idx + 4];
        if (conf < conf_thre) continue;
        x = static_cast<int>(detection_ptr[g_idx]);
        y = static_cast<int>(detection_ptr[g_idx + 1]);
        x1 = static_cast<int>(detection_ptr[g_idx + 2]);
        y1 = static_cast<int>(detection_ptr[g_idx + 3]);
        cls_id = static_cast<int>(detection_ptr[g_idx + 5]);
        w = x1 - x;
        h = y1 - y;

        auto it = class_names.find(cls_id);
        if (it != class_names.end()) {
            index = std::distance(class_names.begin(), it);
        } else {
            std::cout << "predicted class id : " << cls_id << " is wrong" << std::endl;
        }

        // bbox
        cv::Rect rect(x, y, w, h);
        auto color_type = color_table[index % color_table.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << conf;
        std::string str_conf = oss.str();
        std::string text = std::to_string(d_idx) + " " + class_names.at(cls_id) + " " + str_conf;
        cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5, color, 1, 0);

        // print to console 
        std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
        std::to_string(index) + ", " + std::to_string(conf) + ", " + class_names.at(cls_id);
        std::cout << consol_text << std::endl;
    }
}