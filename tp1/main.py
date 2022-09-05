import cv2 as cv


def main():
    circleToCompare = getContoursByShape('./circle6.png', 100)
    squareToCompare = getContoursByShape('./square3.png', 100)
    triangleToCompare = getContoursByShape('./triangle5.png', 127)

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, 255, (lambda a: None))

    while True:
        # 1 - Get original image
        #originalImage = cv.imread("./templateimg4.png")
        ret, originalImage = webcam.read()
        # cv.imshow('Original image', originalImage)

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos('Trackbar', 'Binary')
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow('Binary', binaryImage)

        # 3 - Remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # kernel = structural element
        opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        denoisedImage = closing
        # cv.imshow("Denoised", denoisedImage)

        # 4 - Get contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # 5 - Filter contours
        contours = [c for c in contours if cv.contourArea(c) > 500]
        contours.pop(0)
        #3cv.drawContours(image=originalImage, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        for c in contours:
            if cv.matchShapes(squareToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.03:
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Square', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)
            elif cv.matchShapes(c, triangleToCompare, cv.CONTOURS_MATCH_I2, 0) < 0.75:
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Triangle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)
            elif cv.matchShapes(circleToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.03:
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Circle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)

        cv.imshow('Original image', originalImage)

        key = cv.waitKey(30)
        if key == 27:
            break


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


main()
cv.destroyAllWindows()
