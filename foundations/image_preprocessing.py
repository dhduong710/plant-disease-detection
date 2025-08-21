import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, width, height):
    return cv2.resize(img, (width, height))

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def edge_detection(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)

def plot_images(images, titles, cmap=None):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if cmap:
            plt.imshow(images[i], cmap=cmap)
        else:
            plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    path = "foundations/raw/leaf1.jpg"  
    img = load_image(path)
    resized = resize_image(img, 300, 300)
    gray = convert_to_gray(resized)
    blurred = apply_blur(gray)
    edges = edge_detection(blurred)

    plot_images(
        [resized, gray, blurred, edges],
        ["Resized", "Grayscale", "Blurred", "Edges"],
        cmap="gray",
    )

#python foundations/image_preprocessing.py --input_path leaf1.jpg --output_dir out --size 224 --backend cv2 --save_resized --save_gray --show
