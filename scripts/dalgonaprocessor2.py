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
circle_color = (29, 168, 248)  # dark yellow
cutting_boundary_color = (29, 168, 248)  # darker yellow
small_triangle_color = (84, 236, 251)  # light yellow
circle_radius = 150  # Radius of the circle
milestone_circle_radius = 10  # Radius of the milestone circles
max_fail_count = 100 # Maximum number of failed points
canvas_height = 200  # Height of the black canvas
canvas_size = 150 # Height of the black canvas
# Transparency level (adjust as needed)
border_thickness = 2  # Thickness of the circle border
alpha = 0.5
iswon= False
isloss = False
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

#
# # Function to draw a circle at a specified point
# def draw_circle_at_point(frame, center, radius=10, color=(255, 0, 0)):
#     cv2.circle(frame, center, radius, color, -1)  # Draw filled circle with specified color


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

    # # Draw a blue circle at all three vertices using the new function
    # draw_circle_at_point(frame, point1)
    # draw_circle_at_point(frame, point2)
    # draw_circle_at_point(frame, point3)

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

# Function to add a black canvas of 200 pixels on all sides of the frame
def add_black_canvas(frame, canvas_size=150, canvas_color=(0, 0, 0,0)):
    (h, w) = frame.shape[:2]
    canvas = np.zeros((h + 2 * canvas_size, w + 2 * canvas_size, 4), dtype=np.uint8)
    canvas[:] = (canvas_color[0], canvas_color[1], canvas_color[2], canvas_color[3])  # Set canvas color with alpha


    canvas[canvas_size:canvas_size + h, canvas_size:canvas_size + w] = frame

    # Draw a white border around the camera frame
    border_thickness = 5
    cv2.rectangle(canvas, (canvas_size - border_thickness, canvas_size - border_thickness),
                  (canvas_size + w + border_thickness, canvas_size + h + border_thickness),
                  (255, 255, 255), border_thickness)

    # Calculate the position for centering the heading text
    heading_text = "Dalgona Cookie Cutter"
    heading_font_scale = 1.3
    heading_thickness = 3
    (heading_width, heading_height), _ = cv2.getTextSize(heading_text, cv2.FONT_HERSHEY_SIMPLEX, heading_font_scale,
                                                         heading_thickness)
    heading_x = (canvas.shape[1] - heading_width) // 2
    heading_y = canvas_size - 70

    # Add heading text
    cv2.putText(canvas, heading_text, (heading_x, heading_y), cv2.FONT_HERSHEY_SIMPLEX, heading_font_scale,
                (84, 236, 251), heading_thickness, cv2.LINE_AA)

    # Add instructions at the bottom
    instruction_font_scale = 0.9
    instruction_thickness = 2
    instruction_color = (255, 255, 255)
    instruction_margin = 20 # Margin between instructions and frame

    # Position of the first instruction
    instruction1_y = canvas_size + h + 40 + instruction_margin
    cv2.putText(canvas, "Backspace  :  To Clear", (canvas_size + 10, instruction1_y),
                cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, instruction_color, instruction_thickness, cv2.LINE_AA)

    # Position of the second instruction
    instruction2_y = instruction1_y + instruction_margin
    cv2.putText(canvas, "Enter  :  To Start", (canvas_size + 10, instruction2_y+40),
                cv2.FONT_HERSHEY_SIMPLEX, instruction_font_scale, instruction_color, instruction_thickness, cv2.LINE_AA)

    return canvas

def display_result(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Resize the image to the desired size
    img = cv2.resize(img, (1500, 1300))
    # Create a new window with the given name
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img)
    # Wait indefinitely until a key is pressed
    cv2.waitKey(3000)
    # Destroy the result window
    cv2.destroyWindow("Result")

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
                    cv2.line(frame, tracked_points[i - 1], tracked_points[i], (0, 255, 0), 10)

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

    # Create an RGBA frame with transparent background
    height, width = frame.shape[:2]
    rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)

    # Create a mask for the yellow circle
    circle_center = (width // 2, height // 2)  # Center of the frame
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circle_mask, circle_center, circle_radius, (255), thickness=-1)  # Filled circle
    # Create the yellow circle with alpha channel
    rgba_frame[circle_mask == 255] = (10, 140, 230, 255)  # Yellow with full opacity
    # cv2.circle(frame, circle_center, circle_radius, circle_color,  10)  # Draw circle with thickness of 2

    # Create a mask for the red triangle
    triangle_mask = np.zeros((height, width), dtype=np.uint8)
    triangle_contour = np.array(triangle_large(frame, 150), np.int32)
    cv2.fillPoly(triangle_mask, [triangle_contour], (255))  # Fill triangle mask with white

    # Apply the triangle mask to make the triangle area transparent
    rgba_frame[triangle_mask == 255] = (100, 100, 100, 0)  # Transparent where the triangle is



    # Check the tracked points
    if tracking_started:
        fail_count = 0
        for point in tracked_points:
            if not is_point_in_triangle(point, large_triangle_points) or is_point_in_triangle(point, small_triangle_points):
                fail_count += 1

        if fail_count > max_fail_count:
            print("Player loses the game!")
            isloss= True
            tracking_started = False
            tracking_enabled = False
            tracked_points = []  # Clear tracked points when the game is lost
            newly_tracked_points = []  # Clear newly tracked points
            milestones_crossed.clear()  # Clear milestones when the game is lost
            first_milestone = None  # Reset first milestone
            all_milestones_crossed_flag = False  # Reset flag
            milestones_after_flag.clear()  # Clear milestones after flag is set
            break

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
                iswon= True
                has_won = True
                tracking_started = False
                tracking_enabled = False
                tracked_points = []  # Clear tracked points when the player wins
                newly_tracked_points = []  # Clear newly tracked points
                milestones_crossed.clear()  # Clear milestones when the player wins
                milestones_after_flag.clear()  # Clear milestones after flag is set
                first_milestone = None  # Reset first milestone
                all_milestones_crossed_flag = False  # Reset flag
                break

    # Print the flag value and both sets
    print(f"All milestones crossed flag: {all_milestones_crossed_flag}")
    print(f"Milestones crossed: {milestones_crossed}")
    print(f"Milestones after flag: {milestones_after_flag}")

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Add the black overlay to achieve the translucent effect
    black_overlay = np.zeros_like(frame, dtype=np.uint8)
    blurred_overlay = cv2.GaussianBlur(black_overlay, (21, 21), 0)
    translucent_black_frame = cv2.addWeighted(frame, 1 - alpha, blurred_overlay, alpha, 0)
    rgba_translucent_black_frame = cv2.cvtColor(translucent_black_frame, cv2.COLOR_BGR2BGRA)

    # Combine the transparent frame with the translucent black frame
    final_rgba_frame = cv2.add(rgba_frame, rgba_translucent_black_frame)

    # Add black canvas around the frame
    frame_with_canvas = add_black_canvas(final_rgba_frame, canvas_size, canvas_color=(13, 2, 29, 0))

    # Get frame dimensions
    height, width = frame_with_canvas.shape[:2]
    print("Height :" , height, "Width : ",width)
    # Display the resulting frame
    cv2.imshow("Dalgona Cookie Cutter Game", frame_with_canvas)
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


cv2.destroyWindow("Dalgona Cookie Cutter Game")
if iswon:

    display_result("C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\win.png")  # Display win image
elif isloss:
    display_result("C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\die.png")  # Display win image
else:
    print("Error Occured :] ")

cap.release()
cv2.destroyAllWindows()