from joblib import dump, load
import cv2 as cv
import numpy as np

# Loads the decision tree model


def main():
    classifier = load('./model.joblib')

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, 255, (lambda a: None))
    while True:

        # 1 - Get original image
        originalImage = cv.imread('../tp1/templateimg4.png')
        #ret, originalImage = webcam.read()
        cv.imshow('Original image', originalImage)

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos('Trackbar', 'Binary')
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow('Binary', binaryImage)

        # 3 - Remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # kernel = structural element
        opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        denoisedImage = closing
        debug = originalImage
        cv.imshow("debug", denoisedImage)
        # 4 - Get contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 5 - Filter contours
        contours = [c for c in contours if cv.contourArea(c) > 2000]  # 2000 is an arbitrary number
        contours.pop(0)  # Remove the contour of the image

        cv.drawContours(image=debug, contours=contours, contourIdx=-1, color=(255, 0, 255), thickness=3)
        #cv.imshow("debug", debug)

        for c in contours:
            moments = cv.moments(c)
            huMoments = cv.HuMoments(moments)
            #predictedLabel = classifier.predict([list(moments.values())])
            predictedLabel = classifier.predict(flatten(huMoments))
            print(flatten(huMoments))

        key = cv.waitKey(30)
        if key == 27:
            break

def flatten(_list):
    return list(np.concatenate(_list).flat)

def getBinaryImage(image, value):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh = cv.threshold(grayImage, value, 255, cv.THRESH_BINARY)
    return thresh

def getContoursByShape(imageRoute, threshBottom):
    shape = cv.imread(imageRoute)
    grayShape = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    ret, shapeThresh = cv.threshold(grayShape, threshBottom, 255, cv.THRESH_BINARY_INV)
    shapeContours, hierarchy = cv.findContours(shapeThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image=shape, contours=shapeContours, contourIdx=-1, color=(255, 255, 0), thickness=3)
    return shapeContours[0]


def calculate_hu_moments(grayscale_image):
    """
    :param grayscale_image: A grayscale image from which the hu moments will be calculated
    :return: The hu moments of the given image
    """
    _, binary_image = cv.threshold(grayscale_image, 128, 255, cv.THRESH_BINARY)
    moments = cv.moments(binary_image)
    return cv.HuMoments(moments)




main()
cv.destroyAllWindows()