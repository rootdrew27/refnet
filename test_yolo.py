from ultralytics import YOLO

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('TKAgg')
    
    model = YOLO('yolov8n.pt')
    model.train(data='C:\\Users\\rooty\\OU Research\\RefNet\\RefNet\\RefNet\\datasets\\FOD_og_size\\data.yaml', batch=1, imgsz=(1536,2048))           