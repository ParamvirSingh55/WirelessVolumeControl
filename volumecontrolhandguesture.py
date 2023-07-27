import cv2
import time
import mediapipe as mp

from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Open the camera capture
cap = cv2.VideoCapture(0)

# Initialize variables
pTime = 0
vol = 0
volBar = 400
volPer = 0

# Initialize Mediapipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Get the audio device and set up the volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# Start an infinite loop to process video frames
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Convert the frame from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands to detect hands and landmarks
    results = hands.process(imgRGB)

    # Initialize a list to store the landmark information
    lmList = []

    # Check if hands are detected in the frame
    if results.multi_hand_landmarks:
        # Loop through all the detected hands
        for handLms in results.multi_hand_landmarks:
            # Loop through all the landmarks of the current hand
            for id, lm in enumerate(handLms.landmark):
                # Get the coordinates of the landmark in pixels
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Store the landmark ID and its pixel coordinates in the lmList
                lmList.append([id, cx, cy])
            # Draw landmarks and hand connections on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Check if there are any landmarks detected
    if lmList:
        # Get the coordinates of the thumb tip and the index finger tip
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

        # Draw circles at the thumb tip and index finger tip
        cv2.circle(img, (x1, y1), 4, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 255), cv2.FILLED)

        # Draw a line between the thumb tip and index finger tip
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw a rectangle to represent the volume control area
        cv2.rectangle(img, (50, 150), (85, 400), (225, 0, 255), 3)

        # Update the volume bar based on the hand position
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 255), cv2.FILLED)

        # Display the volume percentage on the image
        cv2.putText(img, f'{int(volPer)}% volume', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 3)

        # Calculate the length between thumb tip and index finger tip
        length = hypot(x2 - x1, y2 - y1)

        # Map the length to a volume value within the volume range
        vol = np.interp(length, [25, 150], [volMin, volMax])
        volBar = np.interp(length, [15, 220], [400, 150])
        volPer = np.interp(length, [15, 220], [0, 100])
        print(vol, length)

        # Set the system volume based on the calculated volume value
        volume.SetMasterVolumeLevel(vol, None)

    # Calculate and display the frames per second (FPS) on the image
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Show the processed image with annotations
    cv2.imshow('Image', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
