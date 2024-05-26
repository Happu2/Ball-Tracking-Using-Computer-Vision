import cv2
import numpy as np
import time

# Function to track balls based on color
def track_balls(frame, color_lower, color_upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tracked_balls = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust min area as needed
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                tracked_balls.append((cx, cy))
    return tracked_balls

# Function to detect entry or exit from quadrants
def detect_entry_exit(previous_positions, current_positions, quadrant_boundaries):
    entry_exit_data = []
    if previous_positions:
        for quadrant_number, (x1, y1, x2, y2) in enumerate(quadrant_boundaries, start=1):
            balls_in_quadrant = [ball for ball in current_positions if x1 <= ball[0] <= x2 and y1 <= ball[1] <= y2]
            prev_balls_in_quadrant = previous_positions.get(quadrant_number, [])
            entering_balls = [ball for ball in balls_in_quadrant if ball not in prev_balls_in_quadrant]
            exiting_balls = [ball for ball in prev_balls_in_quadrant if ball not in balls_in_quadrant]
            entry_exit_data.extend([(time.time(), quadrant_number, 'Color', 'Entry') for _ in entering_balls])
            entry_exit_data.extend([(time.time(), quadrant_number, 'Color', 'Exit') for _ in exiting_balls])
    return entry_exit_data

# Main function
def main():
    video_path = "C:\\Users\\Admin\\PycharmProjects\\pythonProject1\\004\\004.mp4"
    cap = cv2.VideoCapture(video_path)

    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([120, 255, 255])

    quadrant_boundaries = [
        (0, 0, 160, 120),
        (160, 0, 320, 120),
        (320, 0, 480, 120),
        (480, 0, 640, 120)
    ]

    out = cv2.VideoWriter('C:\\Users\\Admin\\PycharmProjects\\pythonProject1\\output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
    output_file = open('C:\\Users\\Admin\\PycharmProjects\\pythonProject1\\output.txt', 'w')

    start_time = time.time()
    previous_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        blue_balls = track_balls(frame, blue_lower, blue_upper)
        all_balls = blue_balls

        entry_exit_data = detect_entry_exit(previous_positions, all_balls, quadrant_boundaries)

        for data in entry_exit_data:
            output_file.write(f"{data[0] - start_time}, Quadrant {data[1]}, {data[2]}, {data[3]}\n")

        for ball in all_balls:
            cv2.circle(frame, ball, 10, (0, 255, 0), -1)
        
        for i, (x1, y1, x2, y2) in enumerate(quadrant_boundaries, start=1):
            cv2.putText(frame, f"Quadrant {i}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        out.write(frame)
        cv2.imshow('Frame', frame)

        previous_positions = {i: [ball for ball in all_balls if quadrant_boundaries[i-1][0] <= ball[0] <= quadrant_boundaries[i-1][2] and quadrant_boundaries[i-1][1] <= ball[1] <= quadrant_boundaries[i-1][3]] for i in range(1, 5)}

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    output_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
