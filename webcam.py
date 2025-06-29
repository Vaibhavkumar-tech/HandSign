import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 300
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        handType = hand['type']
        # Draw hand type ("Right" or "Left") on the image
        cv2.putText(img, handType, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    #extra
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Ensure cropping does not go out of bounds #extra
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgcrop = img[y1:y2, x1:x2]

        # Only resize if the crop is valid
        if imgcrop.size != 0:
            imgcrop_resized = cv2.resize(imgcrop, (imgsize, imgsize))
            imgwhite[0:imgsize, 0:imgsize] = imgcrop_resized
            cv2.imshow("Cropped Image", imgcrop)
            cv2.imshow("White Image", imgwhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)