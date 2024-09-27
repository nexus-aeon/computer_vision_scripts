# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Prints a menu
def print_menu() -> None:
    print("Options menu:")
    print("esc - Quit program")
    print("s - Snapshot")


# Get a BGR image from a VideoCapture device.
def get_image(cam: cv2.VideoCapture, res_scale: float) -> np.ndarray:
    _, img = cam.read()

    # rescale the input image if it's too large
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)
    return img


# Draw and return image at a region of interest.
def get_region_of_interest(img: np.ndarray) -> np.ndarray:

    roi_start_x, roi_start_y, width, height = cv2.selectROI(img)

    roi_end_x = int(roi_start_x + width)
    roi_end_y = int(roi_start_y + height)
    roi_start_x = int(roi_start_x)
    roi_start_y = int(roi_start_y)

    img_roi = img[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    return img_roi


# Draw the color histogram for each channel
def plot_histogram(
    img: np.ndarray, color_space_name: str, channel_names: list[str]
) -> None:
    plt_colors = ["b", "g", "r"]

    plt.figure()
    for i, col in enumerate(channel_names):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=plt_colors[i])
        plt.xlim([0, 256])
        plt.xticks(np.arange(0, 256, 10))
    plt.legend(channel_names)
    plt.title(f"{color_space_name} Histogram")


# Calculates the histogram for an image and a given color space
def calculate_histograms(bgr_img: np.ndarray, color_space_names: list[str]) -> None:

    for colr_space in color_space_names:
        if colr_space == "BGR":
            channel_names = ["Blue", "Green", "Red"]
            plot_histogram(bgr_img, colr_space, channel_names)
        elif colr_space == "HSV":
            channel_names = ["Hue", "Saturation", "Value"]
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            plot_histogram(hsv_img, colr_space, channel_names)
        elif colr_space == "YCrCb":
            channel_names = ["Luminance", "Chrominance-Red", "Chrominance-Blue"]
            ycrcb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
            plot_histogram(ycrcb_img, colr_space, channel_names)
        else:
            cv2.destroyAllWindows()
            sys.exit("Color Space Not Supported!")
    plt.show()


def main() -> None:
    print_menu()

    res_scale = 1.0  # Scale resolution
    wait_time_in_ms = 10  # Sleep time for capturing input
    cam = cv2.VideoCapture(0)
    color_space_names = [
        "BGR",
        "HSV",
        "YCrCb",
    ]  # Color Spaces for Histogram Calculation

    while True:
        img_roi = None
        img = get_image(cam, res_scale)
        cv2.imshow("CAM Preview", img)

        action = cv2.waitKey(wait_time_in_ms)

        if action == 27:
            break
        elif action == ord("h"):
            print_menu()
        elif action == ord("s"):
            img_roi = get_region_of_interest(img)
            cv2.imshow("ROI", img_roi)

        if img_roi is not None:
            calculate_histograms(img_roi, color_space_names)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
