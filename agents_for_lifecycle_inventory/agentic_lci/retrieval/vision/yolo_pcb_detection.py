# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Identify and crop PCB using custom YOLO from https://github.com/SanderGi/PCB-Detection

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
from ultralytics.models.yolo import YOLO

dirname = os.path.dirname(__file__)
pcb_obb = YOLO(os.path.join(dirname, "..", "..", "..", ".models", "pcb_obb_yolov11.pt"))
pcb_seg = YOLO(os.path.join(dirname, "..", "..", "..", ".models", "pcb_seg_yolov11.pt"))


def contains_pcb_confidence(image, model="seg") -> float:
    if model == "seg":
        results = pcb_seg.predict(image)[0]
        return results.boxes.conf[0].item() if len(results.boxes.conf) > 0 else 0  # type: ignore
    elif model == "obb":
        results = pcb_obb.predict(image)[0]
        return results.obb.conf[0].item() if len(results.obb.conf) > 0 else 0  # type: ignore
    else:
        raise ValueError("Model must be either 'seg' or 'obb'")


def contains_pcb(image, model="seg", conf=0.5) -> bool:
    return contains_pcb_confidence(image, model) >= conf


def detect_pcb_oriented_bounding_boxes(image, save_path: str | None = None):
    results = pcb_obb.predict(image)[0]
    if save_path is not None:
        results.save(save_path)
    return results.obb


def detect_pcb_axis_aligned_bounding_boxes(image, save_path: str | None = None):
    results = pcb_seg.predict(image)[0]
    if save_path is not None:
        results.save(save_path)
    return results.boxes


def detect_pcb_masks(image, save_path: str | None = None):
    results = pcb_seg.predict(image)[0]
    if save_path is not None:
        results.save(save_path)
    return results.masks


if __name__ == "__main__":
    # example/test usage
    print(
        detect_pcb_masks(
            os.path.join(
                dirname, "..", "..", "..", "data", "iPhone14ProMaxInternal.jpeg"
            ),
            save_path="test.jpg",
        )
    )
    print(
        contains_pcb(
            os.path.join(dirname, "..", "..", "..", "data", "iPhone13Internal.png")
        )
    )
