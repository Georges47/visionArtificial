import csv

import cv2 as cv
import numpy as np


def main():
    images = [
        cv.imread('tp1/circle6.png', cv.IMREAD_GRAYSCALE),
        cv.imread('tp1/circle3.jpg', cv.IMREAD_GRAYSCALE),
        cv.imread('tp1/square.jpg', cv.IMREAD_GRAYSCALE),
        cv.imread('tp1/square3.png', cv.IMREAD_GRAYSCALE),
        cv.imread('tp1/triangle5.png', cv.IMREAD_GRAYSCALE)
    ]
    labels = [
        0,  # circle
        0,
        1,  # square
        1,
        2   # triangle
    ]
    dataset = generate_dataset(images)
    dataset = dataset_with_labels(dataset, labels)
    save_dataset(dataset)


def generate_dataset(grayscale_images):
    """
    :param grayscale_images: A list of grayscale images from which the dataset will be generated
    :return: An array of vectors, with each vector having the hu moments of an image
    """
    dataset = []
    for image in grayscale_images:
        hu_moments = calculate_hu_moments(image)
        hu_moments = flatten(hu_moments)
        dataset.append(hu_moments)
    return dataset


def calculate_hu_moments(grayscale_image):
    """
    :param grayscale_image: A grayscale image from which the hu moments will be calculated
    :return: The hu moments of the given image
    """
    _, binary_image = cv.threshold(grayscale_image, 128, 255, cv.THRESH_BINARY)
    moments = cv.moments(binary_image)
    return cv.HuMoments(moments)


def flatten(_list):
    return list(np.concatenate(_list).flat)


def dataset_with_labels(dataset, labels):
    """
    :param dataset: dataset without labels, only the hu moments
    :param labels: list of labels for the rows, must have the same order as the dataset
    :return: dataset with hu moments and label
    """
    result_dataset = []
    for row, label in zip(dataset, labels):
        result_row = [row, [label]]
        result_row = flatten(result_row)
        result_dataset.append(result_row)
    return result_dataset


def save_dataset(dataset):
    with open('./tp2/dataset.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'label'])
        writer.writerows(dataset)


def test():
    # 1 Read image as grayscale image
    grayscaleImage = cv.imread('tp1/circle6.png', cv.IMREAD_GRAYSCALE)
    cv.imshow('Grayscale image', grayscaleImage)

    # 2 Create binary image from grayscale image
    _, binaryImage = cv.threshold(grayscaleImage, 128, 255, cv.THRESH_BINARY)
    cv.imshow('Binary image', binaryImage)

    # 3 Calculate Hu moments
    moments = cv.moments(binaryImage)
    huMoments = cv.HuMoments(moments)

    print(huMoments)


main()
