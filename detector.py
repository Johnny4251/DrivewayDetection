import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import nms
from datetime import datetime

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def setup_model(classes_to_detect):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    model.eval()

    class_indices = [COCO_INSTANCE_CATEGORY_NAMES.index(class_name) for class_name in classes_to_detect]

    return device, model, class_indices

def get_time():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime

def capture_stream(classes_to_detect, camera_index=0, use_roi = True, flip=False, score_thresh=0.40, iou_thresh=0.3):
    camera_video = cv2.VideoCapture(camera_index)
    device, model, class_indices = setup_model(classes_to_detect)
    prev_car_count = 0
    while True:
        person_count, car_count = 0, 0
        _, frame = camera_video.read()
        if flip: frame = cv2.flip(frame, 1)
        cv2.imshow('Camera View', frame)

        if use_roi:
            driveway_xmin, driveway_xmax = 200,550
            driveway_ymin, driveway_ymax = 100,500
            driveway_roi = frame[driveway_ymin:driveway_ymax, driveway_xmin:driveway_xmax]
            frame = driveway_roi

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tensor_image = F.to_tensor(frame_rgb)
        tensor_image = tensor_image.to(device)
        
        with torch.no_grad():
            predictions = model([tensor_image])
        for class_index in class_indices:
            class_boxes = [predictions[0]['boxes'][i] for i, label in enumerate(predictions[0]['labels']) if label.item() == class_index]

            if len(class_boxes) > 0:
                boxes = torch.stack(class_boxes, dim=0)
                scores = predictions[0]['scores'][predictions[0]['labels'] == class_index]

                keep = nms(boxes, scores, iou_threshold=iou_thresh)

                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]
                for box, score in zip(filtered_boxes, filtered_scores):
                    if score >= score_thresh:
                        

                        xmin, ymin, xmax, ymax = map(int, box.tolist())
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, f"{classes_to_detect[class_indices.index(class_index)]} {score:.2f}"
                                    , (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if classes_to_detect[class_indices.index(class_index)] == 'person':
                            person_count += 1

                            scale_factor = 30
                            try:
                                zoom_frame = frame[ymin-scale_factor:ymax+scale_factor, xmin-scale_factor:xmax+scale_factor]
                                zoom_frame = cv2.resize(zoom_frame, (480, 640), interpolation=cv2.INTER_LINEAR)

                                cv2.imwrite("person/zoom/"+get_time()+".png", zoom_frame)
                                cv2.imwrite("person/full_photo/"+get_time()+".png", frame)
                                print("Photo taken")
                            except:
                                print("ERROR TO CAPTURE PHOTO: " + get_time())
                        elif classes_to_detect[class_indices.index(class_index)] == 'car':
                            car_count += 1
                            

        cv2.putText(frame, f"Persons: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Cars: {car_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
        if prev_car_count > car_count:
            print("A car has left the driveway")
        elif prev_car_count < car_count:
            print("A car has pulled into the driveway")
            
        prev_car_count = car_count   
        cv2.imshow('Driveway', driveway_roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
            
    camera_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_stream(['person', 'car'], score_thresh=0.61)