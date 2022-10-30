import cv2 as cv


def main():
    circleToCompare = getContoursByShape('./tp1/555.png', 100)
    squareToCompare = getContoursByShape('./tp1/square3.png', 100)
    triangleToCompare = getContoursByShape('./tp1/triangle.png', 100)

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, 255, (lambda a: None))

    while True:
        # 1 - Get original image
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
        contours = [c for c in contours if cv.contourArea(c) > 1000]
        if len(contours) > 0:
            contours.pop(0)
        # cv.drawContours(image=originalImage, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
        print(hierarchy)

        contours_to_draw = []
        for c in contours:
            if cv.matchShapes(squareToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.03:
                # cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                contours_to_draw.append(c)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Square', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)
            elif cv.matchShapes(circleToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.03:
                # cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                contours_to_draw.append(c)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Circle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)
            elif cv.matchShapes(triangleToCompare, c, cv.CONTOURS_MATCH_I2, 0) < 0.08:
                print(cv.matchShapes(triangleToCompare, c, cv.CONTOURS_MATCH_I2, 0))
                # cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)
                contours_to_draw.append(c)
                x, y, w, h = cv.boundingRect(c)
                cv.putText(originalImage, 'Triangle', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                           cv.LINE_AA)

        for c in contours_to_draw:
            cv.drawContours(image=originalImage, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=3)

        cv.imshow('Original image', originalImage)

        key = cv.waitKey(30)
        if key == 27:
            break


def getBinaryImage(image, value):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh = cv.threshold(grayImage, value, 255, cv.THRESH_BINARY)
    return thresh


def getContoursByShape(image_route, thresh_bottom):
    shape = cv.imread(image_route)
    grayShape = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    ret, shapeThresh = cv.threshold(grayShape, thresh_bottom, 255, cv.THRESH_BINARY_INV)
    shapeContours, hierarchy = cv.findContours(shapeThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image=shape, contours=shapeContours, contourIdx=-1, color=(255, 255, 0), thickness=3)
    return shapeContours[0]


main()
cv.destroyAllWindows()
