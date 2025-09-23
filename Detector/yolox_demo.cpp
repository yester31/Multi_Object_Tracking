#include "yolox_opti_trt.hpp"
#include "utils.hpp"

#include <chrono>
#include <cmath>

void draw_bbox_text(
    float* detection_ptr,
    int num_dets,
    cv::Mat &img, 
    float ratio, float conf_thre, const std::string &img_name, const std::string &save_dir_path, 
    const std::vector<std::vector<int>> &color_table, const std::vector<std::string> &class_names)
{
    int img_width = img.cols;
    // int img_hight = img.rows;
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
        auto color_type = COLOR_TABLE[cls_id % COLOR_TABLE.size()];
        auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
        rectangle(img, rect, color, 2, 8, 0);

        // text box
        std::string text = std::to_string(d_idx) + " " + COCO_LABELS[cls_id] + " " + std::to_string(conf);
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        int text_box_x = (x >= (img_width - 250)) ? x - 250 : x;
        int text_box_y = (y < 50) ? y + 50 : y;
        cv::Rect text_box(text_box_x, text_box_y - 30, text_size.width + 10, text_size.height + 15);
        cv::rectangle(img, text_box, color, cv::FILLED);
        cv::putText(img, text, cv::Point(text_box_x, text_box_y - 3), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

        // print to console 
        std::string consol_text = "[" + std::to_string(d_idx) + "] " + img_name + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(w) + ", " + std::to_string(h) + ", " +
        std::to_string(cls_id) + ", " + std::to_string(conf) + ", " + COCO_LABELS[cls_id];
        std::cout << consol_text << std::endl;
    }
}

int main()
{
    std::filesystem::path CUR_DIR = std::filesystem::current_path();
    std::cout << "Current path: " << CUR_DIR << std::endl;
    if (CUR_DIR.filename() == "build") {
        CUR_DIR = CUR_DIR.parent_path();  // 상위 디렉토리로 이동
    }

    // 1) parameter setting
    const int BATCH_SIZE{ 1 };
    const int INPUT_H{ 640 };
    const int INPUT_W{ 640 };
    const int INPUT_C{ 3 };
    const int CLASS_COUNT{ 80 };
    const int precision_mode{ 16 }; // fp32 mode : 32, fp16 mode : 16
    int gpu_device{ 0 };            // gpu device index (default = 0)
    bool serialize{ false };        // force serialize flag (IF true, recreate the engine file unconditionally)
    std::string engine_file_name{ "yolox-s" };  // engine file name (engine file will be generated uisng this name)
    std::filesystem::path engine_dir_path = CUR_DIR / "engine" ;// engine directory path (engine file will be generated in this location)
    std::filesystem::path weight_file_path = CUR_DIR / "onnx/yolox-s_640x640_sim_w_nms.onnx" ; // weight file path

    yolox_opti_trt yolox_trt = yolox_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

    // 2) prepare input data
    std::filesystem::path image_dir_path = CUR_DIR / "data" ; // image file directory path
    std::vector<std::string> image_file_names;
    load_images_from_folder(image_file_names, image_dir_path.string());
    int num_test_imgs = static_cast<int>(image_file_names.size());
    std::cout << "num_test_imgs : "<< num_test_imgs << std::endl;

    std::filesystem::path save_dir_path = CUR_DIR / "results" ; // save file directory path
    gen_dir(save_dir_path.string());

    // 3) Inference results check
    std::vector<cv::Mat> imgs; // temporary image save for visualization
    std::vector<float> ratios; // temporary ratios
    int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
    int OUTPUT_SIZE = (1 + 6 * 300);
    std::vector<uint8_t> inputs0(BATCH_SIZE * INPUT_SIZE);  // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> inputs(BATCH_SIZE * INPUT_SIZE);     // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);   // [BATCH_SIZE, (the number of detection,  {bbox[x,y,w,h], score, cls_id} * 300)]

    for (int i = 0; i < static_cast<int>(ceil(static_cast<float>(num_test_imgs) / BATCH_SIZE)); i++) // batch unit loop
    {
        // load image
        for (int b_idx = 0; b_idx < BATCH_SIZE; b_idx++)
        {
            int imd_idx = (i * BATCH_SIZE + b_idx < num_test_imgs) ? i * BATCH_SIZE + b_idx : num_test_imgs - 1;
            cv::Mat ori_img = cv::imread(image_dir_path.string() + "/" + image_file_names[imd_idx]);
            imgs.push_back(ori_img);
            if (!ori_img.data)
            {
                std::cerr << "[ERROR] Data load error (Check image path)" << std::endl;
            }
            // preprocess input images
            float ratio = std::min((float)INPUT_W / (ori_img.cols), (float)INPUT_H / (ori_img.rows));
            ratios.push_back(ratio);
            pre_proc_yolox_0(inputs0, ratio, ori_img, b_idx, INPUT_SIZE, INPUT_H, INPUT_W);
        }
        pre_proc_yolox_1(inputs, inputs0, BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W); // int8 BGR[NHWC](0, 255) -> float BGR[NCHW](0, 255)

        yolox_trt.input_data(inputs.data());
        yolox_trt.run_model();
        yolox_trt.output_data(outputs.data());

        // draw results
        for (int b_idx = 0; b_idx < BATCH_SIZE; b_idx++)
        {
            float* detection_ptr = outputs.data() + b_idx * OUTPUT_SIZE + 1;
            int num_dets = static_cast<int>(outputs[b_idx * OUTPUT_SIZE + 0]);  // number of detections
            int imd_idx = (i * BATCH_SIZE + b_idx < num_test_imgs) ? i * BATCH_SIZE + b_idx : num_test_imgs - 1;
            cv::Mat img = imgs[imd_idx];
            float ratio = ratios[imd_idx];
            std::string img_name = std::filesystem::path(image_file_names[imd_idx]).stem().string();
            float conf_thre = 0.5;
            draw_bbox_text(detection_ptr, num_dets, img, ratio, conf_thre, img_name, save_dir_path, COLOR_TABLE, COCO_LABELS);

            // show
            // cv::resize(img, img, cv::Size(static_cast<int>(img.cols), static_cast<int>(img.rows)));
            cv::namedWindow(img_name);
            cv::moveWindow(img_name, 30, 30);
            cv::imshow(img_name, img);
            cv::waitKey(0);
            cv::destroyAllWindows();

            // save
            std::string save_file_path = save_dir_path.string() + "/" + img_name + "_trt.jpg";
            std::cout << save_file_path << std::endl;
            cv::imwrite(save_file_path, img); 

            if (!cv::imwrite(save_file_path, img)) {
                std::cerr << "Failed to save file!" << std::endl;
                return -1;
            }

        }
        std::cout << "==========================================================================" << std::endl;
    }

    
    return 0;
}