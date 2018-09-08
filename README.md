# YOLOv3-pytorch
A YOLOv3 model in pytorch0.4.1

usage: 

python detect_one.py --image=images/dog-cycle-car.png --cuda

### Acknowledgements

[Original tech report of YOLOv3 by Joseph Redmon](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

Inspired by [Ayoosh Kathuria](https://github.com/ayooshkathuria)'s amazing tutorial on YOLOv3 implementation in pytorch,

I transformed the model into caffe and implemented a simple [caffe-based yolov3 workflow](https://github.com/jasonlovescoding/YOLOv3-caffe).

This project is its pytorch adaptation.

The weight files can be downloaded from [YOLOv3-pytorch](https://download.csdn.net/download/jason_ranger/10654561).

For people outside China, you can download from googledrive [YOLOv3-pytorch](https://drive.google.com/open?id=1T3oDa5iUH-yJ3VltgqN9wiZNzXb9hAdX)

### Notes

A simplest YOLOv3 model in pytorch0.4.1

This is merely a practice project. I re-implemented the darknet file for readability and 

make modification on the architecture for fine-tuning or transfer-learning easy, which is the main purpose of this project.