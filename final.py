
import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('hgrmodel_pre.h5')

model.summary()

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Nothing']

cap = cv2.VideoCapture(0)

def nothing(x):
    print(x)

cv2.namedWindow('roi')
cv2.createTrackbar('converter', 'roi', 0, 255, nothing)
cv2.createTrackbar("LH", "roi", 0, 255, nothing)
cv2.createTrackbar("LS", "roi", 0, 255, nothing)
cv2.createTrackbar("LV", "roi", 0, 255, nothing)
cv2.createTrackbar("UH", "roi", 255, 255, nothing)
cv2.createTrackbar("US", "roi", 255, 255, nothing)
cv2.createTrackbar("UV", "roi", 255, 255, nothing)

while (True):

    _, frame= cap.read()
    cv2.rectangle(frame, (0, 0), (300, 300), (0, 0, 255), 5)
    # region of interest
    roi = frame[0:300, 0:300]
    cv2.imshow('roi', roi)
    # print(roi.shape)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "roi")
    l_s = cv2.getTrackbarPos("LS", "roi")
    l_v = cv2.getTrackbarPos("LV", "roi")

    u_h = cv2.getTrackbarPos("UH", "roi")
    u_s = cv2.getTrackbarPos("US", "roi")
    u_v = cv2.getTrackbarPos("UV", "roi")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)

    hsvframe = cv2.bitwise_and(roi, roi, mask=mask)
    cv2.imshow('hsvframe', hsvframe)

    grayhsv = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # print(grayhsv.shape)
    blurhsv= cv2.GaussianBlur(grayhsv, (5, 5), 2)

    bConv= cv2.getTrackbarPos('converter', 'roi')
    _, convertedImage= cv2.threshold(grayhsv, bConv, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('converted image', convertedImage)
    convertedImage = cv2.merge([convertedImage, convertedImage, convertedImage])
    finalImage = cv2.resize(convertedImage, (50, 50))
    # print("converted image shape: "+ str(convertedImage.shape))
    # print("final image shape: "+ str(finalImage.shape))


    # make predication about the current frame
    prediction= model.predict(finalImage.reshape(1, 50, 50, 3))

    char_index = np.argmax(prediction)
    # print(char_index, prediction[0, char_index]*100)

    predicted_char = labels[char_index]


    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    cv2.putText(frame, predicted_char, (80, 80), font, fontScale, color, thickness)

    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
