from Detector import *
from cvzone.PoseModule import PoseDetector

def main():
    # 0 to webcan
    # videoPath = 0;
    videoPath = "video/5738706-hd_1920_1080_24fps.mp4";

    modelPath = "frozen_inference_graph.pb"
    configPath ="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    classesPath = "coco.names"

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()