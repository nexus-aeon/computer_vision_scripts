import os
import cv2
import numpy as np


# Read a list of images from a data directory
def read_images(image_names: list[str], data_dir: str) -> list[np.ndarray]:
    img_list: list[np.ndarray] = list()
    for img_name in image_names:
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path)
        img_list.append(img)
    return img_list

# Histogram equalization on individual color channels and convertion to greyscale.
def split_histogram_equalization(image_list: list[np.ndarray]) -> list[np.ndarray]:
    img_eq_list: list[np.ndarray] = list()

    for img in image_list:
        b, g, r = cv2.split(img)

        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        avg_eq_img = np.round((b_eq + g_eq + r_eq) / 3.0).astype(np.uint8)
        img_eq_list.append(avg_eq_img)

    return img_eq_list

# Histogram equalization on the averaged image.
def split_histogram_equalization_alt(image_list: list[np.ndarray]) -> list[np.ndarray]:
    img_eq_list: list[np.ndarray] = list()

    for img in image_list:
        b, g, r = cv2.split(img)
        avg_img = np.round((b + g + r) / 3.0).astype(np.uint8)
        avg_eq_img = cv2.equalizeHist(avg_img)
        img_eq_list.append(avg_eq_img)

    return img_eq_list

# Write a list of images to an output directory.
def save_images(
    image_list: list[np.ndarray], image_names: list[str], img_dir: str
) -> None:
    list_len = len(image_names)

    for idx in range(list_len):
        img = image_list[idx]
        img_name = image_names[idx]
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, img)


def main():
    # Imange names index.
    img_lower_idx = 1
    img_upper_idx = 5

    # Source and Target image directories.
    data_dir = os.path.join("..", "data", "histogram_equalization")
    out_dir = os.path.join("..", "out", "histogram_equalization")
    os.makedirs(out_dir, exist_ok=True)

    img_names = [f"img_{idx}.jpg" for idx in range(img_lower_idx, img_upper_idx + 1)]
    out_names = [
        f"img_out_{idx}.jpg" for idx in range(img_lower_idx, img_upper_idx + 1)
    ]
    alt_out_names = [
        f"img_out_{idx}_alt.jpg" for idx in range(img_lower_idx, img_upper_idx + 1)
    ]

    img_list = read_images(img_names, data_dir)
    img_eq_list = split_histogram_equalization(img_list)
    alt_img_eq_list = split_histogram_equalization_alt(img_list)

    save_images(img_eq_list, out_names, out_dir)
    save_images(alt_img_eq_list, alt_out_names, out_dir)


if __name__ == "__main__":
    main()
