# -*- coding: utf-8 -*-

# Import necessary modules
import cv2
import numpy as np
import time


# Configuration for pre-processing the masks.
def get_morphological_operator_config() -> dict[str, dict[str, int]]:
    closing_config = {"kernel_size": 13, "iter_count": 3}

    dilate_config = {"kernel_size": 11, "iter_count": 1}

    morph_operator_config = {"closing": closing_config, "dilate": dilate_config}

    return morph_operator_config


# Configuration of the color thresholds for various color spaces.
def get_color_thresh(clr_space: str) -> list[tuple[int]]:
    # Thresholds for a sky blue cloak.
    hsv_thresh = [(91, 105), (64, 120), (120, 190)]
    ycrcb_thresh = [(92, 164), (101, 115), (136, 150)]

    clr_space_thresh = {"HSV": hsv_thresh, "YCrCb": ycrcb_thresh}

    return clr_space_thresh[clr_space]


# Convert BGR to various color spaces.
def convert_color_space(bgr_img: np.ndarray, clr_space: str) -> np.ndarray:
    if clr_space == "BGR":
        img = bgr_img
    elif clr_space == "HSV":
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    elif clr_space == "YCrCb":
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

    return img


# Get a BGR image from a VideoCapture device.
def get_image(cam: cv2.VideoCapture, res_scale: float) -> np.ndarray:
    _, img = cam.read()

    # Rescale the input image if it's too large
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)
    return img


# Iterates until a background is obtained.
def get_background(cam: cv2.VideoCapture, res_scale: float) -> np.ndarray:
    background = None
    cv2_ver = cv2.__version__.split(".")[0]

    while cam.isOpened():
        bgr = get_image(cam, res_scale)
        cv2.imshow("Background", bgr)

        action = cv2.waitKey(1)
        if action == ord("s"):
            background = bgr
            break

        # Closing the windows selects the last frame as the background
        if cv2_ver == "3":
            if cv2.getWindowProperty("Background", 0) == -1:
                background = bgr
                break
        if cv2_ver == "4":
            if cv2.getWindowProperty("Background", cv2.WND_PROP_VISIBLE) <= 0:
                background = bgr
                break

    return background


# Creates and pre-process the color based masks for segmentation.
def get_color_mask(
    img: np.ndarray,
    thresh: list[tuple[int]],
    morph_op_config: dict[str, dict[str, int]],
) -> np.ndarray:
    clr_lwr_thresh = np.array([channel_thresh[0] for channel_thresh in thresh])
    clr_upr_thresh = np.array([channel_thresh[1] for channel_thresh in thresh])

    clr_mask = cv2.inRange(img, clr_lwr_thresh, clr_upr_thresh)

    # Apply Closing Morphological Operator
    kernel_size = morph_op_config["closing"]["kernel_size"]
    iter_count = morph_op_config["closing"]["iter_count"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clr_mask = cv2.morphologyEx(
        clr_mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=iter_count
    )

    # Apply Dilate Morphological Operator
    kernel_size = morph_op_config["dilate"]["kernel_size"]
    iter_count = morph_op_config["dilate"]["iter_count"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clr_mask = cv2.morphologyEx(
        clr_mask, cv2.MORPH_DILATE, kernel=kernel, iterations=iter_count
    )

    return clr_mask


# Perform segmentation and blend the images.
def make_clr_invisible(
    background_img: np.ndarray, img: np.ndarray, clr_mask: np.ndarray
) -> np.ndarray:

    inverted_clr_mask = cv2.bitwise_not(
        clr_mask
    )  # Mask which allows every color except the one in our clr_mask.

    res_background_img = cv2.bitwise_and(background_img, background_img, mask=clr_mask)
    res_img = cv2.bitwise_and(img, img, mask=inverted_clr_mask)

    blended_img = cv2.addWeighted(
        res_background_img, 1.0, res_img, 1.0, 0
    )  # Blend two portions toghether.
    return blended_img


def main() -> None:
    res_scale = 1.0  # Scale resolution
    clr_space = "HSV"
    wait_time_in_ms = 10  # Sleep time for capturing input

    cam = cv2.VideoCapture(0)
    time.sleep(3)  # Initial Setup Time in ms

    clr_thresh = get_color_thresh(clr_space)
    morphological_op_config = get_morphological_operator_config()

    background_img = get_background(cam, res_scale)

    while cam.isOpened():
        img = get_image(cam, res_scale)
        clr_space_img = convert_color_space(img, clr_space)
        clr_mask = get_color_mask(clr_space_img, clr_thresh, morphological_op_config)
        cv2.imshow("Final Mask", clr_mask)

        invicible_img = make_clr_invisible(background_img, img, clr_mask)
        cv2.imshow("Magic Frame", invicible_img)

        action = cv2.waitKey(wait_time_in_ms)
        if action & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
