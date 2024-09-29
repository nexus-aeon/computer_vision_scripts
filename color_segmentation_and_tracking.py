# -*- coding: utf-8 -*-

import cv2
import numpy as np

# Get the correct thresholds for given color space
def get_color_thresh(
    clr_names: list[str], clr_space: str
) -> dict[str, list[tuple[int]]]:

    bgr_thresh = {
        "green": [(35, 85), (90, 170), (0, 25)],
        "blue": [(135, 210), (25, 90), (0, 25)],
        "yellow": [(15, 120), (128, 236), (142, 237)],
    }

    hsv_thresh = {
        "green": [(68, 82), (210, 255), (50, 200)],
        "blue": [(106, 120), (220, 255), (120, 220)],
        "yellow": [(24, 34), (125, 220), (140, 240)],
    }

    ycrcb_thresh = {
        "green": [(46, 150), (55, 95), (108, 130)],
        "blue": [(30, 130), (78, 116), (172, 210)],
        "yellow": [(135, 225), (134, 146), (45, 85)],
    }

    clr_space_thresh = {"BGR": bgr_thresh, "HSV": hsv_thresh, "YCrCb": ycrcb_thresh}

    clr_thresh = dict()
    cur_thresh = clr_space_thresh[clr_space]
    for clr_name in clr_names:
        clr_thresh[clr_name] = cur_thresh[clr_name]

    return clr_thresh


# Get a BGR image from a VideoCapture device.
def get_image(cam: cv2.VideoCapture, res_scale: float) -> np.ndarray:
    _, img = cam.read()

    # rescale the input image if it's too large
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)
    return img


# Convert BGR to various color spaces.
def convert_color_space(bgr_img: np.ndarray, clr_space: str) -> np.ndarray:
    if clr_space == "BGR":
        img = bgr_img
    elif clr_space == "HSV":
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    elif clr_space == "YCrCb":
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

    return img


# Creates and pre-process the color based masks for segmentation.
def get_color_mask(
    img: np.ndarray, thresh: list[tuple[int]], kernel_size: int
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clr_lwr_thresh = np.array([channel_thresh[0] for channel_thresh in thresh])
    clr_upr_thresh = np.array([channel_thresh[1] for channel_thresh in thresh])

    clr_mask = cv2.inRange(img, clr_lwr_thresh, clr_upr_thresh)
    # Apply Closing and Dilate Morphological Operators
    clr_mask = cv2.morphologyEx(clr_mask, cv2.MORPH_CLOSE, kernel=kernel)
    clr_mask = cv2.morphologyEx(clr_mask, cv2.MORPH_DILATE, kernel=kernel)

    return clr_mask

# Detect the contours with a minimum size and return its coordinates 
def get_contour_coords(img: np.ndarray, min_object_size: int) -> list[tuple[int]]:
    filtered_contour_coords = list()
    # Find connected components
    _, labels = cv2.connectedComponents(img)
    connected_img = labels.astype(np.uint8)

    # Find contours of these objects
    contours, _ = cv2.findContours(
        connected_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    # Filter out contours with less than minimum size
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_object_size or h > min_object_size:
            filtered_contour_coords.append((x, y, w, h))

    return filtered_contour_coords

# Draw box at detected coordinates with text
def draw_box_and_text(img: np.ndarray, contour_coords: tuple[int], text: str) -> None:

    x, y, w, h = contour_coords

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        img,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    res_scale = 1.0  # Scale resolution
    min_object_size = 30  # Minimum pixels for object
    kernel_size = 7  # Dilation and closing kernel size
    wait_time_in_ms = 10  # Sleep time for capturing input
    cam = cv2.VideoCapture(0)
    clr_space = "YCrCb"
    clr_names = ["green", "blue", "yellow"]

    clr_thresh = get_color_thresh(clr_names, clr_space)

    while True:
        bgr_img = get_image(cam, res_scale)
        img = convert_color_space(bgr_img, clr_space)

        for cur_clr in clr_names:
            obj_idx = 1
            cur_thresh = clr_thresh[cur_clr]
            clr_mask = get_color_mask(img, cur_thresh, kernel_size)
            clr_contour_coords = get_contour_coords(clr_mask, min_object_size)

            for contour_coord in clr_contour_coords:
                obj_text = f"{cur_clr} Object #{obj_idx}"
                draw_box_and_text(bgr_img, contour_coord, obj_text)
                obj_idx += 1

        cv2.imshow("Live WebCam", bgr_img)

        action = cv2.waitKey(wait_time_in_ms)
        if action & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
