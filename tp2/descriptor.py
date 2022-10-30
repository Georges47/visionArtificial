import glob
from math import log10, copysign
import csv
import cv2 as cv


def main():
    dataset = []
    for element in ['square', 'star', 'triangle']:
        for path in glob.glob(f'tp2/images/{element}/*.png'):
            original_image = cv.imread(path)

            gray_image = cv.cvtColor(original_image, cv.COLOR_RGB2GRAY)
            _, binary_image = cv.threshold(gray_image, 45, 255, cv.THRESH_BINARY)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
            opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
            closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
            denoised_image = closing

            contours, hierarchy = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            cv.drawContours(original_image, contours[1], -1, (0, 255, 0), 7)
            cv.imshow(element, original_image)
            cv.waitKey(0)
            moments = cv.moments(contours[1])

            hu_moments = cv.HuMoments(moments)

            row = []
            for j in range(0, 7):
                if hu_moments[j] != 0:
                    hu_moments[j] = -1 * copysign(1.0, hu_moments[j]) * log10(abs(hu_moments[j]))
                    row.append(hu_moments[j][0])

            dataset.append(row)

    with open('tp2/csv/dataset.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataset)


main()
