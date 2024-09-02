from Detector import *

def main():
    videoPath = 0;

    modelPath = "frozen_inference_graph.pb"
    configPath ="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    classesPath = "coco.names"

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()


if __name__ == '__main__':
    main()