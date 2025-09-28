#include "yolo_opti_trt.hpp"
#include "utils.hpp"
#include <chrono>
#include <cmath>

void yolox_demo()
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
    std::filesystem::path weight_file_path = CUR_DIR / "../ONNX_Generator/YOLOX/onnx/yolox-s_640x640_sim_w_nms.onnx" ; // weight file path

    yolo_opti_trt yolox_trt = yolo_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

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
            pre_proc_yolox(inputs, ori_img, ratio, b_idx, INPUT_SIZE, INPUT_H, INPUT_W);
        }

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
            draw_bbox_text_yolox(detection_ptr, num_dets, img, ratio, conf_thre, img_name, save_dir_path, COLOR_TABLE, COCO_LABELS);

            // show
            show_image(img, engine_file_name);

            // save
            std::string save_file_path = save_dir_path.string() + "/" + img_name + "_" + engine_file_name + "_trt.jpg";
            std::cout << save_file_path << std::endl;
            cv::imwrite(save_file_path, img); 

            if (!cv::imwrite(save_file_path, img)) {
                std::cerr << "Failed to save file!" << std::endl;
            }
        }
        std::cout << "==========================================================================" << std::endl;
    }

}