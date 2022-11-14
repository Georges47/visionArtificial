import cv2 as cv
from pip._vendor.distlib.compat import raw_input
import numpy as np


def main():
    image = captureimage()
    grabcut(image)



def captureimage ():
    camera = cv.VideoCapture(0)
    raw_input('Press Enter to capture')
    return_value, image = camera.read()
    return image

def grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # usamos roi para agarrar el rect
    rect = cv.selectROI("img", image, fromCenter=False, showCrosshair=True)
    print(mask)

    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

    print(mask)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    print(mask2)

    image = image*mask2[:, :, np.newaxis]

    cv.imshow("img", image)
    cv.waitKey()


main()
cv.destroyAllWindows()