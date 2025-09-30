#include "OCSort.hpp" 
#include "yolo_opti_trt.hpp"
#include "utils.hpp"

/**
@brief Convert Vector to Matrix
@param data
@return Eigen::Matrix<float, Eigen::Dynamic, 6>
*/
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

template<typename AnyCls>
std::ostream& operator<<(std::ostream& os, const std::vector<AnyCls>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}

void ocsort_yolox_bytetrack_demo() {
    std::filesystem::path CUR_DIR = std::filesystem::current_path();
    std::cout << "Current path: " << CUR_DIR << std::endl;
    if (CUR_DIR.filename() == "build") {
        CUR_DIR = CUR_DIR.parent_path(); 
    }

    // 1) parameter setting
    const int BATCH_SIZE{ 1 };
    const int INPUT_H{ 608 };
    const int INPUT_W{ 1088 };
    const int INPUT_C{ 3 };
    const int CLASS_COUNT{ 1 };
    const int precision_mode{ 16 }; // fp32 mode : 32, fp16 mode : 16
    int gpu_device{ 0 };            // gpu device index (default = 0)
    bool serialize{ false };        // force serialize flag (IF true, recreate the engine file unconditionally)
    std::string engine_file_name{ "bytetrack_s_mot17" };  // engine file name (engine file will be generated uisng this name)
    std::filesystem::path engine_dir_path = CUR_DIR / "../Detector/engine" ;// engine directory path (engine file will be generated in this location)
    std::filesystem::path weight_file_path = CUR_DIR / "../ONNX_Generator/YOLOX_ByteTrack/onnx/bytetrack_s_mot17_608x1088_sim_w_nms.onnx" ; // weight file path

    yolo_opti_trt yolox_bt_trt = yolo_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

    int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
    int OUTPUT_SIZE = (1 + 6 * 300);
    std::vector<uint8_t> inputs0(BATCH_SIZE * INPUT_SIZE);  // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> inputs(BATCH_SIZE * INPUT_SIZE);     // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);   // [BATCH_SIZE, (the number of detection,  {bbox[x,y,w,h], score, cls_id} * 300)]
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std  = {0.229f, 0.224f, 0.225f};

    std::filesystem::path video_file_path = CUR_DIR / "../data/video/palace.mp4"; // video file path
    std::string input_video_path = video_file_path.string();
    cv::VideoCapture cap(input_video_path);
	if (!cap.isOpened())
		std::cerr << "Error opening video stream or file" << std::endl;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;

    std::filesystem::path save_dir_path = CUR_DIR / "results" ; // save file directory path
    gen_dir(save_dir_path.string());
    std::filesystem::path save_file_path = save_dir_path / "ocsort_yolox_bytetrack_demo.mp4" ;
    VideoWriter writer(save_file_path.string(), VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    cv::Mat img;
    ocsort::OCSort tracker = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);
    int num_frames = 0;
    int total_ms = 0;
    float ratio = std::min(static_cast<float>(INPUT_W) / (img_w), static_cast<float>(INPUT_H )/ (img_h));
	while (true)
    {
        if(!cap.read(img))
            break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            std::cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << std::endl;
        }
		if (img.empty())
			break;

        pre_proc_yolox_bt(inputs, img, ratio, 0, INPUT_SIZE, INPUT_H, INPUT_W, mean, std);

        // run inference
        auto start = chrono::system_clock::now();
        yolox_bt_trt.input_data(inputs.data());
        yolox_bt_trt.run_model();
        yolox_bt_trt.output_data(outputs.data());

        int x, y, x1, y1;
        float conf;
        int label;
        int num_dets = static_cast<int>(outputs[0]);  // number of detections
        float* detection_ptr = outputs.data() + 1;
        float conf_thre = 0.45;
        std::vector<std::vector<float>> data;
        std::vector<Eigen::RowVectorXf> res;
        for (int d_idx = 0; d_idx < num_dets; d_idx++)
        {   
            int g_idx = d_idx * 6;
            conf = detection_ptr[g_idx + 4];
            if (conf < conf_thre) continue;
            label = static_cast<int>(detection_ptr[g_idx + 5]);
            if (label != 0) continue;
            x = static_cast<int>(detection_ptr[g_idx] / ratio);
            y = static_cast<int>(detection_ptr[g_idx + 1] / ratio);
            x1 = static_cast<int>(detection_ptr[g_idx + 2] / ratio);
            y1 = static_cast<int>(detection_ptr[g_idx + 3] / ratio);
            std::vector<float> row;
            row.push_back(x);
            row.push_back(y);
            row.push_back(x1);
            row.push_back(y1);
            row.push_back(conf);
            row.push_back(label);
            data.push_back(row);
        }

        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();

        if (!data.empty()) {
            res = tracker.update(Vector2Matrix(data));

            for (auto j : res) {
                int ID = int(j[4]);
                int Class = int(j[5]);
                float conf = j[6];
                auto color_type = COLOR_TABLE[ID % COLOR_TABLE.size()];
                auto color = cv::Scalar(color_type[0], color_type[1], color_type[2]);
                cv::putText(img, cv::format("%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                cv::rectangle(img, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), color, 2);
            }
            data.clear();
        }

        putText(img, format("frame: %d fps: %d num: %d", 
            static_cast<int>(num_frames), 
            static_cast<int>(num_frames * 1000000 / total_ms), 
            static_cast<int>(res.size())), Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);

        cv::imshow(save_file_path, img);
        char c = waitKey(1);
        if (c > 0)
        {
            break;
        }
    }

    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;
}