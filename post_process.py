import cv2

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt


def draw_line(img: NDArray[np.uint8], center: tuple[float, float], 
              angle: float, length: int, color: tuple[int, int, int] = (255, 0, 0), thickness: int = 1
              ) -> NDArray[np.uint8]:
    x1 = int(center[0] + length / 2 * np.cos(np.radians(angle)))
    y1 = int(center[1] + length / 2 * np.sin(np.radians(angle)))
    x2 = int(center[0] - length / 2 * np.cos(np.radians(angle)))
    y2 = int(center[1] - length / 2 * np.sin(np.radians(angle)))
    
    start_point = (x1, y1)
    end_point = (x2, y2)

    fig = cv2.line(img, start_point, end_point, color, thickness)
    return fig


def find_centroid_and_orientation(mask: NDArray[np.uint8]) -> float:
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contour, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)

    center, (width, height), angle = rect

    if width < height:
        angle += 90

    return angle, center


def post_process(
    img: NDArray[np.uint8],
    msk: NDArray[np.uint8]
) -> tuple[tuple[float, float], float, NDArray[np.uint8]]:
    mask = np.where(msk>0, 255, 0).astype(np.uint8) # convert mask to binary
    angle, centroid = find_centroid_and_orientation(mask)
    centroid = int(centroid[0]), int(centroid[1])

    overlay = img.copy()
    overlay[mask>0] = (0, 0, 255) # draw the mask in red
    overlay = cv2.circle(overlay, centroid, radius=2, color=(0, 255, 0), thickness=-1) # draw the centroid in green
    overlay = draw_line(overlay, centroid, angle, length=50, color=(255, 0, 0)) # draw the orientation line in blue

    return centroid, angle, overlay


if __name__ == "__main__":
    img = cv2.imread("image001.png").astype(np.uint8)
    mask = cv2.imread("image002.png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    _, _, pred = post_process(img, mask)
    
    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[1].imshow(mask, cmap="gray")
    ax[2].imshow(pred)
    for a in ax:
        a.axis("off")
    plt.show()