import cv2 as cv


def main():
    # square = cv.imread('./square3.png')
    # graySquare = cv.cvtColor(square, cv.COLOR_BGR2GRAY)
    # ret, squareThresh = cv.threshold(graySquare, 100, 255, cv.THRESH_BINARY_INV)
    # squareContours, hierarchy = cv.findContours(squareThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # ##cv.drawContours(image=square, contours=squareContours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    # #cv.imshow('Binary2', square)
    #
    # circle = cv.imread('./circle6.png')
    # grayCircle = cv.cvtColor(circle, cv.COLOR_BGR2GRAY)
    # ret, circleThresh = cv.threshold(grayCircle, 100, 255, cv.THRESH_BINARY_INV)
    # circleContours, hierarchy = cv.findContours(circleThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(image=circle, contours=circleContours, contourIdx=0, color=(0, 255, 0), thickness=3)
    # #cv.imshow('Binary3', circle)
    #
    # triangle = cv.imread('./triangle5.png')
    # grayTriangle = cv.cvtColor(triangle, cv.COLOR_BGR2GRAY)
    # ret, triangleThresh = cv.threshold(grayTriangle, 127, 255, cv.THRESH_BINARY_INV)
    # triangleContours, hierarchy = cv.findContours(triangleThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(image=triangle, contours=triangleContours, contourIdx=-1, color=(0, 255, 0), thickness=3)
    #cv.imshow('Binary4', triangle)

    circleToCompare = getContoursByShape('./circle6.png', 100)
    sqToCompare = getContoursByShape('./square3.png', 100)
    trinToCompare = getContoursByShape('./triangle5.png', 127)

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, 255, (lambda a: None))
    while True:

        # 1 - Get original image
        # originalImage = cv.imread('./templateimg4.png')
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
        debug = originalImage
        # cv.imshow("debug", denoisedImage)
        # 4 - Get contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 5 - Filter contours
        contours = [c for c in contours if cv.contourArea(c) > 2000]  # 2000 is an arbitrary number
        contours.pop(0)  # Remove the contour of the image

        # cv.drawContours(image=debug, contours=contours, contourIdx=-1, color=(255, 0, 255), thickness=3)
        # cv.imshow("debug", debug)

        for c in contours:
            if cv.matchShapes(sqToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.05:  # 0.3 is an arbitrary number
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Square', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
            elif cv.matchShapes(c, trinToCompare, cv.CONTOURS_MATCH_I2, 0) < 0.05:  # 0.3 is an arbitrary number
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Triangle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
            elif cv.matchShapes(circleToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.05:  # 0.3 is an arbitrary number
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Circle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
            else :
                cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 0, 255), thickness=3)

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
