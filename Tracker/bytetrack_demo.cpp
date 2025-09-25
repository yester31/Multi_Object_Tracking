#include "BYTETracker.h" 
#include "yolo_opti_trt.hpp"
#include "yolox_demo.hpp"
#include "utils.hpp"

int main() {
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
    std::filesystem::path engine_dir_path = CUR_DIR / "../Detector/engine" ;// engine directory path (engine file will be generated in this location)
    std::filesystem::path weight_file_path = CUR_DIR / "../Detector/onnx/yolox-s_640x640_sim_w_nms.onnx" ; // weight file path

    yolo_opti_trt yolox_trt = yolo_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

    int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
    int OUTPUT_SIZE = (1 + 6 * 300);
    std::vector<uint8_t> inputs0(BATCH_SIZE * INPUT_SIZE);  // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> inputs(BATCH_SIZE * INPUT_SIZE);     // [BATCH_SIZE, 640, 640, 3]
    std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);   // [BATCH_SIZE, (the number of detection,  {bbox[x,y,w,h], score, cls_id} * 300)]

    std::filesystem::path video_file_path = CUR_DIR / "../data/video/palace.mp4"; // video file path
    std::string input_video_path = video_file_path.string();
    cv::VideoCapture cap(input_video_path);
	if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;

    std::filesystem::path save_file_path = CUR_DIR / "demo.mp4" ;
    VideoWriter writer(save_file_path.string(), VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    cv::Mat img;
    BYTETracker tracker(fps, 30);
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

        pre_proc_yolox_0(inputs0, ratio, img, 0, INPUT_SIZE, INPUT_H, INPUT_W);
        pre_proc_yolox_1(inputs, inputs0, BATCH_SIZE, INPUT_C, INPUT_H, INPUT_W); // int8 BGR[NHWC](0, 255) -> float BGR[NCHW](0, 255)

        // run inference
        auto start = chrono::system_clock::now();
        yolox_trt.input_data(inputs.data());
        yolox_trt.run_model();
        yolox_trt.output_data(outputs.data());
        vector<Object> objects;

        int x, y, x1, y1;
        float conf;
        int num_dets = static_cast<int>(outputs[0]);  // number of detections
        float* detection_ptr = outputs.data() + 1;
        float conf_thre = 0.5;
        objects.resize(num_dets);
        for (int d_idx = 0; d_idx < num_dets; d_idx++)
        {   
            int g_idx = d_idx * 6;
            conf = detection_ptr[g_idx + 4];
            if (conf < conf_thre) continue;
            x = static_cast<int>(detection_ptr[g_idx] / ratio);
            y = static_cast<int>(detection_ptr[g_idx + 1] / ratio);
            x1 = static_cast<int>(detection_ptr[g_idx + 2] / ratio);
            y1 = static_cast<int>(detection_ptr[g_idx + 3] / ratio);
            objects[d_idx].rect.x = x;
            objects[d_idx].rect.y = y;
            objects[d_idx].rect.width = x1 - x;
            objects[d_idx].rect.height = y1 - y;
            objects[d_idx].label = static_cast<int>(detection_ptr[g_idx + 5]);
            objects[d_idx].prob = conf;
        }

        vector<STrack> output_stracks = tracker.update(objects);
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();

        if(false){
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                bool vertical = tlwh[2] / tlwh[3] > 1.6;
                if (tlwh[2] * tlwh[3] > 20 && !vertical)
                {
                    Scalar s = tracker.get_color(output_stracks[i].track_id);
                    putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                            0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                }
            }
        }else{
            for (int i = 0; i < output_stracks.size(); i++)
            {
                vector<float> tlwh = output_stracks[i].tlwh;
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }

        putText(img, format("frame: %d fps: %d num: %d", static_cast<int>(num_frames), static_cast<int>(num_frames * 1000000 / total_ms), 
            static_cast<int>(output_stracks.size())), Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);

        cv::imshow("test", img);
        char c = waitKey(1);
        if (c > 0)
        {
            break;
        }
    }

    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;
    return 0;
}