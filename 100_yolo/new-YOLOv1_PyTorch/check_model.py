
from models.yolo import myYOLO


if __name__ == '__main__':
    import argparse, torchsummary

    #print("\nmyYOLO:")
    model = myYOLO(device='cpu', input_size=(416, 416))
    torchsummary.summary(model, input_size=(3, 416, 416), batch_size=1, device='cpu')