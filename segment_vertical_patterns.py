import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Union


# Returns a configuration for a gabor kernel
def get_gabor_config() -> dict[str:float]:
    gabor_config = {
        "kernel_size": 10.0,  # Gabor filter size
        "sigma": 3.5,  # Width of the gaussian envelope
        "lambda": 2.5,  # Wavelength of the sinusoidal oscillations
        "theta": 0.0,  # Orientation of the filter
    }

    return gabor_config


# Returns a configuration for segmentation processing
def get_segment_config() -> dict[str : Union[int, tuple[int]]]:
    segment_config = {
        "thresh_type": cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,  # Threshold using otsu method
        "morph_ops": (
            cv2.MORPH_CLOSE,
            cv2.MORPH_ERODE,
        ),  # Morphological operations in order of application
        "morph_ops_size": (
            8,
            10,
        ),  # Structuring element size for morphological operations
    }

    return segment_config


# Create a gabor kernel for vertical pattern extraction on greyscale images
def get_gabor_kernel(kernel_config: dict[str:float]) -> np.ndarray:
    gamma = 1.0
    psi = 1.0
    kernel_type = cv2.CV_32F
    sigma = kernel_config["sigma"]
    lmda = kernel_config["lambda"]
    theta = kernel_config["theta"]
    kernel_size = int(kernel_config["kernel_size"])
    kernel_size_tuple = (kernel_size, kernel_size)

    kernel = cv2.getGaborKernel(
        kernel_size_tuple, sigma, theta, lmda, gamma, psi, kernel_type
    )

    # Normalize the kernel and remove DC component
    kernel /= kernel.sum()
    kernel -= kernel.mean()

    return kernel


# Applies a sequence of morphological operators.
def apply_morphological_operations(
    img: np.ndarray, ops: tuple[int], sizes: tuple[int]
) -> np.ndarray:
    assert len(ops) == len(
        sizes
    ), "Configuration error: No of Morphological operations not equal"

    num_ops = len(ops)
    for op_idx in range(num_ops):
        morph_op = ops[op_idx]
        morph_size = sizes[op_idx]

        morph_kernel = np.ones((morph_size, morph_size), np.uint8)
        img = cv2.morphologyEx(img, morph_op, kernel=morph_kernel)

    return img


# Segment vertical patterns in the image using filtering, thresholding and morphological operations
def segment_vertical_patterns(
    img: np.ndarray,
    gabor_kernel: np.ndarray,
    segment_config: dict[str : Union[int, tuple[int]]],
) -> np.ndarray:
    thresh = 0
    img_maxval = 255
    thresh_type = segment_config["thresh_type"]
    morph_ops = segment_config["morph_ops"]
    morph_ops_size = segment_config["morph_ops_size"]
    ddepth = cv2.CV_8UC3  # Destinatin depth for greyscale image

    # Apply the gabor filter
    fltrd_img = cv2.filter2D(img, ddepth, gabor_kernel)

    # Apply thresholding
    _, thresh_img = cv2.threshold(fltrd_img, thresh, img_maxval, thresh_type)

    # Apply morpholical operators
    segmented_img = apply_morphological_operations(
        thresh_img, morph_ops, morph_ops_size
    )

    return segmented_img


# Display a kernel as a 3D surface plot.
def display_kernel(kernel: np.ndarray) -> None:
    max_x_val = kernel.shape[0]
    max_y_val = kernel.shape[1]

    # Define the coordinates of the grid
    xx, yy = np.mgrid[0:max_x_val, 0:max_y_val]

    # Plot the kernel
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(xx, yy, kernel, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    plt.show()


# Save a single image
def save_image(img: np.ndarray, name: str, dest_dir: str) -> None:
    img_path = os.path.join(dest_dir, name)

    os.makedirs(dest_dir, exist_ok=True)
    cv2.imwrite(img_path, img)


# Display a single image
def display_image(img: np.ndarray, name: str) -> None:
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    # Setup the read and write directories
    data_dir = os.path.join("..", "data", "vertical_pattern_segmentation")
    out_dir = os.path.join("..", "out", "vertical_pattern_segmentation")

    # Read the image
    img_name = "pattern.png"
    out_name = "pattern_segmented.png"
    img_path = os.path.join(data_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Segment the vertical patterns using gabor filter
    kernel_config = get_gabor_config()
    segment_config = get_segment_config()
    gabor_kernel = get_gabor_kernel(kernel_config)
    segmented_img = segment_vertical_patterns(img, gabor_kernel, segment_config)

    display_kernel(gabor_kernel)
    save_image(segmented_img, out_name, out_dir)
    display_image(segmented_img, out_name)


if __name__ == "__main__":
    main()
