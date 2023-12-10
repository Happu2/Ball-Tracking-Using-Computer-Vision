import cv2
import numpy as np
import time

# Function to track balls based on color
def track_balls(frame, color_lower, color_upper):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv, color_lower, color_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracked_balls = []

    for contour in contours:
        # Filter contours based on area (adjust min_area based on your needs)
        if cv2.contourArea(contour) > 100:
            # Get the centroid of the contour
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
            # Check for balls entering or exiting each quadrant
            balls_in_quadrant = [ball for ball in current_positions if x1 <= ball[0] <= x2 and y1 <= ball[1] <= y2]

            if not previous_positions.get(quadrant_number):
                # If there were no previous positions, consider all balls as entering
                entry_exit_data.extend([(time.time(), quadrant_number, 'Color', 'Entry') for _ in balls_in_quadrant])
            else:
                # Check for balls entering and exiting
                entering_balls = [ball for ball in balls_in_quadrant if ball not in previous_positions[quadrant_number]]
                exiting_balls = [ball for ball in previous_positions[quadrant_number] if ball not in balls_in_quadrant]

                entry_exit_data.extend([(time.time(), quadrant_number, 'Color', 'Entry') for _ in entering_balls])
                entry_exit_data.extend([(time.time(), quadrant_number, 'Color', 'Exit') for _ in exiting_balls])

    return entry_exit_data

# Main function
def main():
    video_path = "C:\\Users\\Admin\\PycharmProjects\\pythonProject1\\004\\004.mp4"
    cap = cv2.VideoCapture(video_path)

    # Define color ranges for different balls (adjust these based on your video)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([120, 255, 255])

    # Define quadrant boundaries (adjust these based on your video)
    quadrant_boundaries = [
        (0, 0, 100, 240),
        (100, 0, 200, 240),
        (200, 0, 300, 240),
        (300, 0, 400, 240)
    ]

    # Output video settings
    out = cv2.VideoWriter('C:\\Users\\Admin\\PycharmProjects\\pythonProject1\\output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))

    # Output text file
    output_file = open('output.txt', 'w')

    start_time = time.time()
    previous_positions = {}  # Dictionary to store previous ball positions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)

        # Print additional information
        print("Current Frame:", cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("Total Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("FPS:", cap.get(cv2.CAP_PROP_FPS))

        print("Video processing complete.")

        # Your ball tracking logic for different colors
        blue_balls = track_balls(frame, blue_lower, blue_upper)

        # Combine positions of balls of different colors
        all_balls = blue_balls  # Add other colors as needed

        # Your logic to detect entry or exit from quadrants
        entry_exit_data = detect_entry_exit(previous_positions, all_balls, quadrant_boundaries)

        # Write data to text file
        for data in entry_exit_data:
            output_file.write(f"{data[0] - start_time}, Quadrant {data[1]}, {data[2]}, {data[3]}\n")

        # Overlay text on the frame (for visualization purposes)
        for ball in all_balls:
            cv2.circle(frame, ball, 10, (0, 255, 0), -1)  # Draw a circle at the ball position

        for i, (x1, y1, x2, y2) in enumerate(quadrant_boundaries, start=1):
            cv2.putText(frame, f"Quadrant {i}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Write frame to output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

        # Update previous_positions for the next iteration
        previous_positions = {i: ball_positions for i, ball_positions in enumerate([blue_balls], start=1)}

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    output_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
