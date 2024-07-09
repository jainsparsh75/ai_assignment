import cv2
import numpy as np
import time



def get_quadrant(center, frame_width, frame_height):
    x, y = center
    if x < frame_width / 2 and y < frame_height / 2:
        return 1
    elif x >= frame_width / 2 and y < frame_height / 2:
        return 2
    elif x < frame_width / 2 and y >= frame_height / 2:
        return 3
    else:
        return 4



def record_event(events, time, quadrant, color, event_type):
    events.append(f"{time:.2f}, {quadrant}, {color}, {event_type}")



def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for different balls (example ranges for red, green, and blue)
    color_ranges = {
        'red': ((0, 120, 70), (10, 255, 255)),
        'green': ((36, 25, 25), (70, 255, 255)),
        'blue': ((94, 80, 2), (126, 255, 255)),
        'orange': ((10, 100, 100), (25, 255, 255)),
        'white': ((0, 0, 200), (180, 30, 255))
    }

    detected_balls = []

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filter small areas
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                detected_balls.append((color, center, radius))

    return detected_balls


# Loading the video
video_path = r'C:\Users\jains\PycharmProjects\AI_assignment\AI Assignment video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # saving the output video
    output_path = r'C:\Users\jains\PycharmProjects\AI_assignment\Output_video'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Event tracking data
    events = []
    previous_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = current_frame / fps

        balls = detect_balls(frame)

        current_positions = {}
        for color, center, radius in balls:
            quadrant = get_quadrant(center, frame_width, frame_height)
            current_positions[(color, quadrant)] = center

            if (color, quadrant) not in previous_positions:
                record_event(events, timestamp, quadrant, color, 'Entry')
                cv2.putText(frame, f"Entry - {color}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                previous_center = previous_positions[(color, quadrant)]
                if quadrant != get_quadrant(previous_center, frame_width, frame_height):
                    record_event(events, timestamp, quadrant, color, 'Entry')
                    cv2.putText(frame, f"Entry - {color}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_positions[(color, quadrant)] = center

            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.putText(frame, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        for key in previous_positions:
            if key not in current_positions:
                color, quadrant = key
                record_event(events, timestamp, quadrant, color, 'Exit')
                previous_center = previous_positions[key]
                cv2.putText(frame, f"Exit - {color}", previous_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        previous_positions = current_positions

        out.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Saving the events to a text file
    events_path = r'C:\Users\jains\PycharmProjects\AI_assignment\Output_text'
    with open(events_path, 'w') as f:
        for event in events:
            f.write(event + '\n')

    print(f"Processed video saved to {output_path}")
    print(f"Events recorded in {events_path}")
