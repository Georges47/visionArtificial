import cv2 as cv


def main():
    square = cv.imread('tp1/square.jpg')
    graySquare = cv.cvtColor(square, cv.COLOR_BGR2GRAY)
    ret, squareThresh = cv.threshold(graySquare, 0, 255, 0)
    squareContours, hierarchy = cv.findContours(squareThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    webcam = cv.VideoCapture(0)
    cv.namedWindow('Binary')
    cv.createTrackbar('Trackbar', 'Binary', 0, 100, (lambda a: None))
    while True:
        # 1 - Get original image
        ret, originalImage = webcam.read()
        # cv.imshow('Original image', originalImage)

        # 2 - Get binary image
        binaryValue = cv.getTrackbarPos('Trackbar', 'Binary')
        binaryImage = getBinaryImage(originalImage, binaryValue)
        cv.imshow('Binary', binaryImage)

        # 3 - Remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # kernel = structural element
        opening = cv.morphologyEx(binaryImage, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        denoisedImage = closing
        # cv.imshow('Denoised', denoisedImage)

        # 4 - Get contours
        contours, hierarchy = cv.findContours(denoisedImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # 5 - Filter contours
        contours = [c for c in contours if cv.contourArea(c) > 2000]  # 2000 is an arbitrary number
        contours.pop(0)  # Remove the contour of the image
        # if len(contours) > 0:
        #     cv.drawContours(originalImage, contours, -1, (0, 255, 0), 3)
            # cv.imshow('Contours', originalImage)

        # 6, 7 - Compare filtered contours with references (square, triangle, circle) and show image with recognized object annotated
        references = squareContours
        for c in contours:
            for r in references:
                if cv.matchShapes(c, r, cv.CONTOURS_MATCH_I2, 0) < 0.3:  # 0.3 is an arbitrary number
                    cv.drawContours(image=originalImage, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
                    x, y, w, h = cv.boundingRect(c)
                    cv.putText(originalImage, 'Square', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow('Contours', originalImage)

        key = cv.waitKey(30)
        if key == 27:
            break


def getBinaryImage(image, value):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh = cv.threshold(grayImage, value, 255, cv.THRESH_BINARY)
    return thresh


main()
cv.destroyAllWindows()
