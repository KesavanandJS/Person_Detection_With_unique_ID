import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize MultiTracker
multi_tracker = cv2.legacy.MultiTracker_create()

# ID counter for persons
person_id_counter = 1
person_ids = []

def detect_persons(frame):
    """
    Detect persons in a frame using YOLOv5 and return bounding boxes with confidence scores.
    """
    results = model(frame)
    detections = results.pandas().xyxy[0]
    
    # Filter persons based on class label (0 for 'person' in COCO)
    persons = detections[detections['class'] == 0]
    
    # Add a column for confidence score and return the filtered results
    persons['confidence'] = persons['confidence']
    
    return persons

def initialize_trackers(frame, persons):
    """
    Initialize trackers for detected persons and assign unique IDs starting from 1.
    """
    global person_id_counter, person_ids

    for _, row in persons.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Create a legacy tracker for the person
        tracker = cv2.legacy.TrackerCSRT_create()
        multi_tracker.add(tracker, frame, bbox)

        # Assign and store the person ID starting from 1
        person_ids.append(person_id_counter)
        person_id_counter += 1

def main():
    """
    Main function to capture video, detect, and track persons.
    """
    global person_ids

    # Capture video
    video = cv2.VideoCapture(0)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Perform YOLO detection and initialize trackers if no tracking data
        if not person_ids:
            persons = detect_persons(frame)
            
            # Filter persons with a confidence score above 0.5
            persons = persons[persons['confidence'] > 0.5]
            
            initialize_trackers(frame, persons)

        # Update trackers
        success, boxes = multi_tracker.update(frame)

        # Draw bounding boxes, person IDs, and confidence scores
        for i, box in enumerate(boxes):
            x, y, w, h = [int(v) for v in box]
            confidence = persons.iloc[i]['confidence']  # Get confidence for each detected person
            
            # Draw bounding box and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {person_ids[i]} Conf: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Person Tracking with IDs', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
