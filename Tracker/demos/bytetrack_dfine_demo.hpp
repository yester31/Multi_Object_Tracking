#include "BYTETracker.h" 
#include "dfine_demo.hpp"

void bytetrack_dfine_demo() {
    std::filesystem::path CUR_DIR = std::filesystem::current_path();
    std::cout << "Current path: " << CUR_DIR << std::endl;
    if (CUR_DIR.filename() == "build") {
        CUR_DIR = CUR_DIR.parent_path();
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
    std::filesystem::path engine_dir_path = CUR_DIR / "../Detector/engine" ;// engine directory path (engine file will be generated in this location)
    std::filesystem::path weight_file_path = CUR_DIR / "../ONNX_Generator/D-FINE/onnx/dfine_s_obj2coco_640x640_sim.onnx" ; // weight file path

    detr_opti_trt dfine_trt = detr_opti_trt(BATCH_SIZE, INPUT_H, INPUT_W, INPUT_C, CLASS_COUNT, precision_mode, serialize, gpu_device, engine_dir_path.string(), engine_file_name, weight_file_path.string());

    int INPUT_SIZE = INPUT_H * INPUT_W * INPUT_C;
    int OUTPUT_SIZE = (6 * 300);
    std::vector<float> inputs(BATCH_SIZE * (INPUT_SIZE + 2 * 2)); // [BATCH_SIZE, input(640, 640, 3), ori_size(2(int64_t))]
    std::vector<float> outputs(BATCH_SIZE * OUTPUT_SIZE);   // [BATCH_SIZE, (boxes[x,y,w,h], scores, labels) * 300]

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
    std::filesystem::path save_file_path = save_dir_path / "bytetrack_dfine_demo.mp4" ;
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

        pre_proc_dfine(inputs, img, 0, INPUT_SIZE, INPUT_H, INPUT_W);
        std::vector<int64_t> ori_size{static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)};
        memcpy(inputs.data() + INPUT_SIZE, ori_size.data(), 2 * sizeof(int64_t)); 

        // run inference
        auto start = chrono::system_clock::now();
        dfine_trt.input_data(inputs.data());
        dfine_trt.run_model();
        dfine_trt.output_data(outputs.data());
        
        vector<Object> objects;
        int x, y, x1, y1;
        float conf;
        float* detection_ptr = outputs.data();
        float conf_thre = 0.1;
        for (int d_idx = 0; d_idx < 300; d_idx++)
        {   
            int g_idx = d_idx * 6;
            conf = detection_ptr[g_idx + 4];
            if (conf < conf_thre) continue;
            x = static_cast<int>(detection_ptr[g_idx]);
            y = static_cast<int>(detection_ptr[g_idx + 1]);
            x1 = static_cast<int>(detection_ptr[g_idx + 2]);
            y1 = static_cast<int>(detection_ptr[g_idx + 3]);
            Object object;
            object.rect.x = x;
            object.rect.y = y;
            object.rect.width = x1 - x;
            object.rect.height = y1 - y;
            object.label = static_cast<int>(detection_ptr[g_idx + 5]);
            object.prob = conf;
            objects.push_back(object);  
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
}