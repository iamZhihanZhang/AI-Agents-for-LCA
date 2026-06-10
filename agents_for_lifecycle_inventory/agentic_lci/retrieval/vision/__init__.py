# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Internal PCB photos full computer vision pipeline

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from .pcb_detection import sort_by_yolo_confidence
from .yolo_pcb_detection import detect_pcb_axis_aligned_bounding_boxes
from .pcb_bounding_boxes import (
    crop,
    generate_character_boxes,
    generate_edge_boxes,
    select_pcb_with_most_components,
    shrink_bounding_box,
    find_ruler_candidates,
    select_best_ruler_candidate,
    crop_and_rotate_horizontal,
    read_ruler_with_ocr,
)
from .yolo_pcb_segmentation import segment


def extract_components_and_dimensions_from_internal_photos(images):
    image = sort_by_yolo_confidence(images)[0]

    character_boxes, character_image = generate_character_boxes(image)
    contour_boxes, countour_images = generate_edge_boxes(image)
    # NOTE: SAM disabled because it is super slow, this does mean an accuracy loss
    masks = []

    detected_pcb = detect_pcb_axis_aligned_bounding_boxes(image)
    if detected_pcb is not None and len(detected_pcb.xyxy) > 0:
        pcb_box = detected_pcb.xyxy[0]
        pcb_box = (int(pcb_box[0]), int(pcb_box[1])), (int(pcb_box[2]), int(pcb_box[3]))
        just_pcb_image = crop(image, pcb_box[0], pcb_box[1])
    else:
        # fall back to old non custom YOLO method
        pcb_p1, pcb_p2, image_with_pcb_highlighted = select_pcb_with_most_components(
            image, masks, contour_boxes
        )
        pcb_box, just_pcb_image = shrink_bounding_box(pcb_p1, pcb_p2, image)

    candidate_rulers = find_ruler_candidates(
        image, masks, contour_boxes, character_boxes
    )
    ruler_p1, ruler_p2, image_with_ruler_highlighted = select_best_ruler_candidate(
        image, candidate_rulers, pcb_box, character_boxes
    )
    just_ruler_image = crop_and_rotate_horizontal(image, ruler_p1, ruler_p2)

    # NOTE: change this line to determine the pixel to distance ratio using a different method than reading a ruler
    pixelsPerInch = read_ruler_with_ocr(image, just_ruler_image)
    pixelsPerMM = pixelsPerInch / 25.4
    pcb_width, pcb_height = (
        just_pcb_image.shape[1] / pixelsPerMM,
        just_pcb_image.shape[0] / pixelsPerMM,
    )

    components = segment(just_pcb_image)
    component_descriptions = [f"Estimated PCB size: {pcb_width:.2f}x{pcb_height:.2f}mm"]
    for pred in components["predictions"]:
        w, h = (
            int(pred["width"]),
            int(pred["height"]),
        )
        confidence, category = pred["confidence"], pred["class"]
        component_descriptions.append(
            f"{category},{confidence:.2f},{w / pixelsPerMM:.2f},{h / pixelsPerMM:.2f}"
        )

    return (
        component_descriptions,
        image_with_pcb_highlighted,  # full image with the PCB colored in
        just_ruler_image,  # just the ruler so the LLM can verify detection and reading
        components[
            "annotated_image"
        ],  # image cropped to the pcb with component bounding boxes
    )
