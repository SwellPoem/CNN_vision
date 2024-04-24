import torch
import sys
sys.path.append('/Users/vale/Desktop/Sapienza/Vision/newProj/pytorch_YOLOv4_master')
from pytorch_YOLOv4_master.tool import darknet2pytorch

def convert(cfg_path, weights_path, output_path):
    model = darknet2pytorch.Darknet(cfg_path)
    model.load_weights(weights_path)
    torch.save(model.state_dict(), output_path)

convert('yolo_hand_detection_master/models/cross-hands.cfg', 'yolo_hand_detection_master/models/cross-hands.weights', 'cross-hands.pth')