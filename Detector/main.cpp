#include "yolox_demo.hpp"
#include "yolo11_demo.hpp"
#include "dfine_demo.hpp"
#include "yolo12_demo.hpp"
#include "deim_demo.hpp"
#include "rt_detr_demo.hpp"
#include "rf_detr_demo.hpp"


int main()
{
    yolox_demo();
    yolo11_demo();
    dfine_demo();
    yolo12_demo();
    deim_demo();
    rt_detr_demo();
    rf_detr_demo();

    return 0;
}