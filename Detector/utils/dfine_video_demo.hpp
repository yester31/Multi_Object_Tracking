#include "detr_opti_trt.hpp"
#include "utils.hpp"

void dfine_video_demo()
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
    std::string engine_file_name{ "dfine_s_obj2coco" };  // engine file name (engine file will be generated uisng this name)
    std::filesystem::path engine_dir_path = CUR_DIR / "engine" ;// engine directory path (engine file will be generated in this location)
    std::filesystem::path weight_file_path = CUR_DIR / "../ONNX_Generator/D-FINE/onnx/dfine_s_obj2coco_640x640_sim.onnx" ; // weight file path

    detr_opti_trt dfine_trt = detr_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

    // 2) prepare input data
    // std::filesystem::path video_file_path = CUR_DIR / "../data/video/palace.mp4"; // video file path
    std::filesystem::path video_file_path = CUR_DIR / "../data/video/drone_video.mp4"; // video file path
    std::string input_video_path = video_file_path.string();
    cv::VideoCapture cap(input_video_path);
	if (!cap.isOpened())
		std::cerr << "Error opening video stream or file" << std::endl;

	int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps_ori = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;
    std::cout << "Video Resolution : " << img_h << ", "<< img_w <<" (H,W)" << std::endl;

    std::filesystem::path save_dir_path = CUR_DIR / "results" ; // save file directory path
    gen_dir(save_dir_path.string());
    std::string save_video_path = save_dir_path.string() + "/" + engine_file_name + "_video_demo.mp4";
    cv::VideoWriter writer(save_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps_ori, cv::Size(img_w, img_h));


    // 3) Inference results check
    int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
    int OUTPUT_SIZE = (6 * 300);
    std::vector<float> inputs(BATCH_SIZE * (INPUT_SIZE + 2 * 2)); // [BATCH_SIZE, input(640, 640, 3), ori_size(2(int64_t))]
    std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);   // [BATCH_SIZE, (boxes[x,y,w,h], scores, labels) * 300]

    cv::Mat img;
    int64_t num_frames = 0;
    int64_t total_ms = 0;
    double fps;
	while (true){
        if(!cap.read(img))
            break;
        num_frames ++;
        if (num_frames % 50 == 0){
            std::cout << "Processing frame " << num_frames << " (" << fps << " fps)" << std::endl;
        }
		if (img.empty()) break;

        pre_proc_detr(inputs, img, 0, INPUT_SIZE, INPUT_H, INPUT_W);
        std::vector<int64_t> ori_size{static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)};
        memcpy(inputs.data() + INPUT_SIZE, ori_size.data(), 2 * sizeof(int64_t)); 

        // run inference
        auto start = std::chrono::steady_clock::now();

        dfine_trt.input_data(inputs.data());
        dfine_trt.run_model();
        dfine_trt.output_data(outputs.data());

        auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        float* detection_ptr = outputs.data();
        int fin_num_dets = 0;  // number of detections
        float conf_thre = 0.5;
        draw_bbox_text_detr(detection_ptr, img, conf_thre, "", save_dir_path, COLOR_TABLE, COCO_LABELS, false, &fin_num_dets);

        // show
        fps = (num_frames * 1e6) / total_ms;
        putText(img, cv::format("frame: %d / %d fps: %.2f num_dets: %d", 
                static_cast<int>(nFrame),
                static_cast<int>(num_frames), 
                fps, 
                static_cast<int>(fin_num_dets)), 
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::namedWindow(engine_file_name);
        cv::moveWindow(engine_file_name, 30, 30);
        cv::imshow(engine_file_name, img);
        writer.write(img);
        char c = cv::waitKey(1);
        if (c > 0) break;
    }

    cap.release();
    std::cout << "FPS: " << fps << std::endl;
}