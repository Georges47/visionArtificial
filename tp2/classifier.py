import cv2 as cv
from joblib import load
from math import copysign, log10


def main():
    classifier = load('tp2/model.joblib')

    window_name = 'Values'
    cv.namedWindow(window_name)
    cv.createTrackbar('Binary', window_name, 0, 255, (lambda a: None))
    cv.createTrackbar('Denoise', window_name, 0, 255, (lambda a: None))

    webcam = cv.VideoCapture(0)
    while True:
        binary_threshold = int(cv.getTrackbarPos('Binary', window_name) / 2) * 2 + 3
        denoise_radius = int(cv.getTrackbarPos('Denoise', window_name) / 2) * 2 + 3

        # 1 - Get original image
        ret, original_image = webcam.read()

        # 2 - Get binary image
        gray_image = cv.cvtColor(original_image, cv.COLOR_RGB2GRAY)
        ret2, binary_image = cv.threshold(gray_image, binary_threshold, 255, cv.THRESH_BINARY)

        # 3 - Remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (denoise_radius, denoise_radius))
        opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        denoised_image = closing

        # 4 - Get contours
        contours, _ = cv.findContours(denoised_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # 5 - Filter contours
        contours = [c for c in contours if 5000 < cv.contourArea(c) < 20000]

        if len(contours) > 0:
            for c in contours:
                moments = cv.moments(c)
                hu_moments = cv.HuMoments(moments)
                for i in range(len(hu_moments)):
                    if hu_moments[i] != 0:
                        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
                predicted_tag = classifier.predict(hu_moments.reshape(-1, 7))

                label = None
                color = (0, 0, 0)
                if predicted_tag == 0:
                    color = (255, 0, 0)
                    label = 'Square'
                elif predicted_tag == 1:
                    color = (255, 255, 0)
                    label = 'Star'
                elif predicted_tag == 2:
                    color = (255, 0, 255)
                    label = 'Triangle'
                if label is not None:
                    cv.drawContours(original_image, [c], -1, color, 2)
                    x, y, w, h = cv.boundingRect(c)
                    cv.putText(original_image, label, (x, y), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1,
                               cv.LINE_AA)

        cv.imshow('Values', denoised_image)
        cv.imshow('Original image', original_image)

        key = cv.waitKey(1)
        if key == 27:
            break

    webcam.release()


main()
