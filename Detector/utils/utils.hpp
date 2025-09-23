
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

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

std::vector<std::string> COCO_LABELS = {
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

// Do data pre-processing
void pre_proc_yolox_0(    
    std::vector<uint8_t> &inputs0,  // output uint8_t data
    float &ratio,     // output ratio data
    cv::Mat &ori_img,               // input image
    int b_idx, int INPUT_SIZE, int INPUT_H, int INPUT_W)
{
    int unpad_w = ratio * ori_img.cols;
    int unpad_h = ratio * ori_img.rows;

    // resize
    cv::Mat resized_img(unpad_h, unpad_w, CV_8UC3);
    cv::resize(ori_img, resized_img, resized_img.size());

    // pad
    cv::Mat padded_img(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_img.copyTo(padded_img(cv::Rect(0, 0, resized_img.cols, resized_img.rows)));
    
    memcpy(inputs0.data() + b_idx * INPUT_SIZE, padded_img.data, INPUT_SIZE);
}

void pre_proc_yolox_1(
    std::vector<float> &output,     // output float data
    std::vector<uint8_t> &input,    // input uint8_t data
    int BatchSize, int channels, int height, int width)
{
    /*
        INPUT  = BGR[NHWC](0, 255)
        OUTPUT = BGR[NCHW]
        - Shuffle form HWC to CHW
    */
    int offset = channels * height * width;
    int b_off, c_off, h_off, h_off_o;
    for (int b = 0; b < BatchSize; b++)
    {
        b_off = b * offset;
        for (int c = 0; c < channels; c++)
        {
            c_off = c * height * width + b_off;
            for (int h = 0; h < height; h++)
            {
                h_off = h * width + c_off;
                h_off_o = h * width * channels + b_off;
                for (int w = 0; w < width; w++)
                {
                    int dstIdx = h_off + w;
                    int srcIdx = h_off_o + w * channels;
                    output[dstIdx] = (static_cast<const float>(input[srcIdx]));
                }
            }
        }
    }
};
