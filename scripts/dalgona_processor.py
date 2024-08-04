import cv2
import numpy as np
import time
import math

# Initialize Mediapipe Hands and other setups
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define colors and parameters
circle_color = (251, 181, 59)  # RGB for hex #fbb53b
cutting_boundary_color = (0, 0, 255)  # Red color
small_triangle_color = (0, 255, 0)  # Green color
circle_radius = 150  # Radius of the circle
milestone_circle_radius = 10  # Radius of the milestone circles
max_fail_count = 100 # Maximum number of failed points

# Video capture
cap = cv2.VideoCapture(0)

# Check if video capture is opened
if not cap.isOpened():
    raise RuntimeError("Failed to open video capture")

# To keep track of previous point for drawing the line
prev_point = None
tracking_started = False
tracking_enabled = False
last_hand_visible_time = time.time()

# List to store the tracked points
tracked_points = []
newly_tracked_points = []  # New list for points after the flag
fail_count = 0

# Variables to track milestones and winning condition
milestones_crossed = set()
milestones_after_flag = set()
first_milestone = None
all_milestones_crossed_flag = False


# Function to draw a circle at a specified point
def draw_circle_at_point(frame, center, radius=10, color=(255, 0, 0)):
    cv2.circle(frame, center, radius, color, -1)  # Draw filled circle with specified color


# Function to draw the centered green triangle
def triangle_small(frame, size):
    (frame_height, frame_width) = frame.shape[:2]

    # Calculate the height of the triangle using the equilateral triangle properties
    height = int(math.sqrt(3) / 2 * size)

    # Calculate the shift amount
    vertical_shift = 0

    # Calculate the vertices of the triangle
    point1 = (frame_width // 2, (frame_height // 2) - (2 * height // 3) - vertical_shift)  # Top vertex
    point2 = (frame_width // 2 - size // 2, (frame_height // 2) + (height // 3) - vertical_shift)  # Bottom left vertex
    point3 = (frame_width // 2 + size // 2, (frame_height // 2) + (height // 3) - vertical_shift)  # Bottom right vertex

    points = np.array([point1, point2, point3], np.int32)
    cv2.fillPoly(frame, [points], small_triangle_color)

    # Draw a blue circle at all three vertices using the new function
    draw_circle_at_point(frame, point1)
    draw_circle_at_point(frame, point2)
    draw_circle_at_point(frame, point3)

    return points, [point1, point2, point3]


# Function to draw the large red triangle with a cutting boundary
def triangle_large(frame, size):
    (frame_height, frame_width) = frame.shape[:2]

    # Calculate the height of the triangle using the equilateral triangle properties
    height = int(math.sqrt(3) / 2 * size)

    # Calculate the shift amount
    vertical_shift = 0

    # Calculate the vertices of the triangle
    point1 = (frame_width // 2, (frame_height // 2) - (2 * height // 3) - vertical_shift)  # Top vertex
    point2 = (frame_width // 2 - size // 2, (frame_height // 2) + (height // 3) - vertical_shift)  # Bottom left vertex
    point3 = (frame_width // 2 + size // 2, (frame_height // 2) + (height // 3) - vertical_shift)  # Bottom right vertex

    points = np.array([point1, point2, point3], np.int32)
    cv2.fillPoly(frame, [points], cutting_boundary_color)

    # Draw a red cutting boundary around the triangle
    # Increase the size of the cutting boundary triangle by 40 pixels on each side
    boundary_size = size + 40
    boundary_height = int(math.sqrt(3) / 2 * boundary_size)
    boundary_point1 = (frame_width // 2, (frame_height // 2) - (2 * boundary_height // 3) - vertical_shift)
    boundary_point2 = (
        frame_width // 2 - boundary_size // 2, (frame_height // 2) + (boundary_height // 3) - vertical_shift)
    boundary_point3 = (
        frame_width // 2 + boundary_size // 2, (frame_height // 2) + (boundary_height // 3) - vertical_shift)

    boundary_points = np.array([boundary_point1, boundary_point2, boundary_point3], np.int32)
    cv2.polylines(frame, [boundary_points], isClosed=True, color=cutting_boundary_color, thickness=2)

    return boundary_points


# Function to check if a point is inside a triangle
def is_point_in_triangle(point, triangle):
    def area(p1, p2, p3):
        return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)

    def is_point_in_area(p, t):
        A = area(t[0], t[1], t[2])
        A1 = area(p, t[1], t[2])
        A2 = area(t[0], p, t[2])
        A3 = area(t[0], t[1], p)
        return A == A1 + A2 + A3

    return is_point_in_area(point, triangle)


# Function to check if a point is within a circle
def is_point_in_circle(point, center, radius):
    return np.linalg.norm(np.array(point) - np.array(center)) <= radius


# Function to calculate milestones based on traced points
def calculate_milestones(tracked_points, circle_centers, radius):
    milestones_crossed = set()
    for point in tracked_points:
        for i, center in enumerate(circle_centers):
            if is_point_in_circle(point, center, radius):
                milestones_crossed.add(i + 1)
    return milestones_crossed


# Initialize variables
milestones_crossed = set()
milestones_after_flag = set()
first_milestone = None
all_milestones_crossed_flag = False
# Update newly_tracked_points if the flag is True

newly_tracked_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Processing frame with Mediapipe
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw landmarks on frame
    if results.multi_hand_landmarks:
        last_hand_visible_time = time.time()  # Update time when hand is visible
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if tracking_started:
                # Get position of the index finger tip (landmark 8)
                index_finger_tip = landmarks.landmark[8]
                h, w, _ = frame.shape
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)

                # Add the current point to the tracked points list
                tracked_points.append((x, y))
                if all_milestones_crossed_flag:
                    newly_tracked_points.append((x, y))

                # Draw the green line following the forefinger
                for i in range(1, len(tracked_points)):
                    cv2.line(frame, tracked_points[i - 1], tracked_points[i], (0, 255, 0), 2)

    else:
        # Clear tracking data if no hand is visible and tracking is enabled
        if tracking_enabled and time.time() - last_hand_visible_time > 5:
            tracked_points = []
            newly_tracked_points = []  # Clear the new tracked points
            tracking_started = False
            tracking_enabled = False

    # Draw the centered green triangle and get vertices
    small_triangle_points, circle_centers = triangle_small(frame, 150)

    # Draw the large red triangle with a cutting boundary
    large_triangle_points = triangle_large(frame, 150)

    # Draw a circle around the triangle
    circle_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Center of the frame
    cv2.circle(frame, circle_center, circle_radius, circle_color, 2)  # Draw circle with thickness of 2

    # Check the tracked points
    if tracking_started:
        fail_count = 0
        for point in tracked_points:
            if not is_point_in_triangle(point, large_triangle_points) or is_point_in_triangle(point, small_triangle_points):
                fail_count += 1

        if fail_count > max_fail_count:
            print("Player loses the game!")
            tracking_started = False
            tracking_enabled = False
            tracked_points = []  # Clear tracked points when the game is lost
            newly_tracked_points = []  # Clear newly tracked points
            milestones_crossed.clear()  # Clear milestones when the game is lost
            first_milestone = None  # Reset first milestone
            all_milestones_crossed_flag = False  # Reset flag
            milestones_after_flag.clear()  # Clear milestones after flag is set

        # Calculate milestones based on tracked points
        current_milestones = calculate_milestones(tracked_points, circle_centers, milestone_circle_radius)

        if tracking_started:
            # Check if all milestones are crossed
            if len(current_milestones) == 3:
                if first_milestone is None:
                    # Convert the set to a list to access by index
                    milestones_list = list(current_milestones)
                    first_milestone = milestones_list[0]  # Save the first milestone
                current_milestones.clear()  # Reset milestones crossed set for new counting
                all_milestones_crossed_flag = True  # Set the flag to True

            # Update milestones_crossed with current_milestones for future checks
            milestones_crossed.update(current_milestones)



            # Update milestones_after_flag using the newly_tracked_points
            if all_milestones_crossed_flag:
                milestones_after_flag.update(calculate_milestones(newly_tracked_points, circle_centers, milestone_circle_radius))

            # Check if the saved milestone is crossed again
            if first_milestone is not None and first_milestone in milestones_after_flag:
                print("Player wins the game!")
                has_won = True
                tracking_started = False
                tracking_enabled = False
                tracked_points = []  # Clear tracked points when the player wins
                newly_tracked_points = []  # Clear newly tracked points
                milestones_crossed.clear()  # Clear milestones when the player wins
                milestones_after_flag.clear()  # Clear milestones after flag is set
                first_milestone = None  # Reset first milestone
                all_milestones_crossed_flag = False  # Reset flag

    # Print the flag value and both sets
    print(f"All milestones crossed flag: {all_milestones_crossed_flag}")
    print(f"Milestones crossed: {milestones_crossed}")
    print(f"Milestones after flag: {milestones_after_flag}")

    # Display frame (for debugging or real-time visualization)
    cv2.imshow('Dalgona Game', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # Enter key
        tracking_started = True
        tracking_enabled = True
        tracked_points = []  # Start new tracking when Enter is pressed
        newly_tracked_points = []  # Clear the new tracked points list
    elif key == 8:  # Backspace key
        tracked_points = []  # Clear tracked points when Backspace is pressed
        newly_tracked_points = []  # Clear the new tracked points list
        tracking_started = False
        tracking_enabled = False

cap.release()
cv2.destroyAllWindows()
