import mediapipe as mp
import cv2
import numpy as np
import time

# from playsound import playsound  # Import playsound for sound playback
#
# def play_sound(file):
#     playsound(file)  # Play the sound file using playsound

cap = cv2.VideoCapture(0)
cPos = 0
startT = 0
endT = 0
userSum = 0
dur = 0
isAlive = 1
isInit = False
cStart, cEnd = 0, 0
isCinit = False
tempSum = 0
winner = 0
inFrame = 0
inFramecheck = False
thresh = 200


def calc_sum(landmarkList):
    tsum = 0
    for i in range(11, 33):
        tsum += (landmarkList[i].x * 480)
    return tsum


def calc_dist(landmarkList):
    return (landmarkList[28].y * 640 - landmarkList[24].y * 640)


def isVisible(landmarkList):
    if (landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7):
        return True
    return False


def apply_blur_outside_bbox(image, bbox, blur_strength=15):
    # Create a mask with the same dimensions as the image
    mask = np.ones_like(image, dtype=np.uint8) * 255
    x_min, y_min, x_max, y_max = bbox

    # Draw filled rectangle on the mask where the body is located
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

    # Apply Gaussian blur to the entire image
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

    # Combine blurred image with the original image using the mask
    result = np.where(mask == 255, blurred_image, image)

    return result


def draw_bounding_box(image, landmarkList, thickness=5):
    # Collect x and y coordinates of pose landmarks
    x_coords = [int(landmarkList[i].x * image.shape[1]) for i in range(len(landmarkList))]
    y_coords = [int(landmarkList[i].y * image.shape[0]) for i in range(len(landmarkList))]

    # Define the bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Extend the bounding box 200 pixels longer from the top
    y_min = max(y_min - 200, 0)  # Ensure y_min does not go out of bounds

    # Draw a thicker green rectangle around the detected person
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)  # Thicker green rectangle


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

im1 = cv2.imread('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\im1.png')
im2 = cv2.imread('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\im2.png')

# Load the images to display on game end
die_img = cv2.imread('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\die.png')
win_img = cv2.imread('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\win.png')

currWindow = im1

while True:
    _, frm = cap.read()
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    frm = cv2.blur(frm, (5, 5))
    drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if not inFramecheck:
        try:
            if isVisible(res.pose_landmarks.landmark):
                inFrame = 1
                inFramecheck = True
            else:
                inFrame = 0
        except:
            print("You are not visible at all")

    if inFrame == 1:
        if not isInit:
            # play_sound('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\greenLight.mp3')
            currWindow = im1
            startT = time.time()
            endT = startT
            dur = np.random.randint(1, 5)
            isInit = True

        if (endT - startT) <= dur:
            try:
                m = calc_dist(res.pose_landmarks.landmark)
                if m < thresh:
                    cPos += 1
                print("current progress is : ", cPos)
            except:
                print("Not visible")

            endT = time.time()
        else:
            if cPos >= 100:
                print("WINNER")
                winner = 1
            else:
                if not isCinit:
                    isCinit = True
                    cStart = time.time()
                    cEnd = cStart
                    currWindow = im2
                    # play_sound('C:\\Users\\Dell\\PycharmProjects\\Squid Game\\squidgame\\game\\static\\game\\redLight.mp3')
                    # time.sleep(2)  # Wait for 2 seconds after playing the redLight sound
                    userSum = calc_sum(res.pose_landmarks.landmark)

                if (cEnd - cStart) <= 3:
                    tempSum = calc_sum(res.pose_landmarks.landmark)
                    cEnd = time.time()
                    if abs(tempSum - userSum) > 150:
                        print("DEAD ", abs(tempSum - userSum))
                        isAlive = 0
                else:
                    isInit = False
                    isCinit = False

        # Get bounding box coordinates
        if res.pose_landmarks:
            bbox = (0, 0, frm.shape[1], frm.shape[0])
            # Apply blurred effect outside bounding box
            frm = apply_blur_outside_bbox(frm, bbox)
            # Draw bounding box around the body with a thicker line
            draw_bounding_box(frm, res.pose_landmarks.landmark, thickness=5)

        # Display the progress
        cv2.circle(currWindow, ((55 + 6 * cPos), 280), 15, (0, 0, 255), -1)
        mainWin = np.concatenate((cv2.resize(frm, (800, 400)), currWindow), axis=0)
        cv2.imshow("Main Window", mainWin)
    else:
        cv2.putText(frm, "Please Make sure you are fully in frame", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 4)
        cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or isAlive == 0 or winner == 1:
        cv2.destroyAllWindows()
        cap.release()
        break

# Show end game image
if isAlive == 0:
    end_img = die_img
elif winner == 1:
    end_img = win_img
else:
    end_img = frm

# Resize the end image to match the window size
end_img = cv2.resize(end_img, (800, 400))

cv2.imshow("End Window", end_img)

# Wait for 5 seconds (5000 milliseconds) and then close the window
cv2.waitKey(5000)
cv2.destroyAllWindows()
