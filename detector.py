import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import nms

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

def is_in_roi(box, roi):
    xmin, ymin, xmax, ymax = box
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
    return xmin >= roi_xmin and xmax <= roi_xmax and ymin >= roi_ymin and ymax <= roi_ymax

def capture_stream(classes_to_detect, camera_index=0, flip=False, score_thresh=0.40, iou_thresh=0.3):
    camera_video = cv2.VideoCapture(camera_index)
    device, model, class_indices = setup_model(classes_to_detect)

    person_in_driveway = False
    person_in_court = False

    car_in_driveway = False

    while True:
        person_count, car_count = 0, 0
        _, frame = camera_video.read()
        if flip: frame = cv2.flip(frame, 1)

        driveway_xmin, driveway_xmax = 150,500
        driveway_ymin, driveway_ymax = 100,500
        driveway_roi = frame[driveway_ymin:driveway_ymax, driveway_xmin:driveway_xmax]
    
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
                        prev_car_count = car_count
                        if classes_to_detect[class_indices.index(class_index)] == 'person':
                            person_count += 1
                        elif classes_to_detect[class_indices.index(class_index)] == 'car':
                            car_count += 1


                        xmin, ymin, xmax, ymax = map(int, box.tolist())
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(frame, f"{classes_to_detect[class_indices.index(class_index)]} {score:.2f}"
                                    , (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        

        #cv2.putText(frame, f"Persons: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #cv2.putText(frame, f"Cars: {car_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Camera View', frame)
        cv2.imshow('Driveway', driveway_roi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
            
    camera_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_stream(['person', 'car'], score_thresh=0.61)