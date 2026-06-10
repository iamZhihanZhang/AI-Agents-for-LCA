# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Various methods and heuristics for identifying which photo is the PCB from a collection of candidate photos

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import numpy as np
from scipy import stats
from .yolo_pcb_segmentation import segment
from .yolo_pcb_detection import contains_pcb_confidence


# ========================= FFT heuristic =========================
def analyze_image_fft(image):
    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    return magnitude_spectrum


def compare_images(img1, img2):
    # Analyze images using FFT
    mag_spectrum1 = analyze_image_fft(img1)
    mag_spectrum2 = analyze_image_fft(img2)

    # Sum of high frequency components
    high_freq_sum1 = np.sum(mag_spectrum1)
    high_freq_sum2 = np.sum(mag_spectrum2)

    # Determine which image has more high frequencies
    return high_freq_sum1 > high_freq_sum2


def sort_by_high_frequency(images):
    images = images.copy()
    images.sort(key=lambda img: np.sum(analyze_image_fft(img)), reverse=True)
    return images


# ========================= Color heuristic =========================
def background_ratio(img, background_threshold=0.1):
    ratio = (
        np.sum(
            np.all(
                np.abs(img - stats.mode(img.reshape(-1, img.shape[2]), axis=0).mode[0])
                < 255 * background_threshold,
                axis=2,
            )
        )
        / img.size
    )
    if ratio < 0.1:  # handle very noisy backgrounds
        ratio = 1.0
    return ratio


def sort_by_amount_of_background_color(images, background_threshold=0.1):
    images = images.copy()
    images.sort(
        key=lambda img: background_ratio(
            img, background_threshold=background_threshold
        ),
    )
    return images


# ========================= Custom YOLO heuristic =========================
def sort_by_yolo_confidence(images):
    confidences = [(i, contains_pcb_confidence(i)) for i in images]
    confidences.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in confidences]


# ========================= Combine Heuristics =========================
def combine_heuristics(
    images,
    heuristic1=background_ratio,
    heuristic2=lambda img: -np.sum(analyze_image_fft(img)),
):
    images = images.copy()
    h1scores = [heuristic1(img) for img in images]
    h2scores = [heuristic2(img) for img in images]
    normalized_h1scores = (h1scores - np.min(h1scores)) / (
        np.max(h1scores) - np.min(h1scores)
    )
    normalized_h2scores = (h2scores - np.min(h2scores)) / (
        np.max(h2scores) - np.min(h2scores)
    )
    f1_scores = [
        (i * j) / (i + j) for i, j in zip(normalized_h1scores, normalized_h2scores)
    ]
    sort_with_indices = sorted(enumerate(f1_scores), key=lambda x: x[1])
    images = [images[i] for i, _ in sort_with_indices]
    return images


def pick_best_image_by_component_count_and_heuristic(
    images, threshold=25, heuristic=sort_by_amount_of_background_color
):
    # Sort by heuristic
    images = heuristic(images)

    # The correct image will have the most segmented components
    # We stop early if we hit the threshold since this can be slow
    segments = []
    for img in images:
        segs = len(segment(img)["predictions"])
        if segs >= threshold:
            return img
        segments.append(segs)

    # If we didn't hit the threshold, return the image with the most segments
    return images[np.argmax(segments)]


if __name__ == "__main__":
    # example/test usage
    import cv2

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    image_paths = [
        os.path.join(data_dir, "iPhone14ProMaxInternal.jpeg"),
        os.path.join(data_dir, "iPhone13Internal.png"),
    ]
    images = [cv2.imread(i) for i in image_paths]
    cv2.imwrite("test.png", sort_by_yolo_confidence(images)[0])
