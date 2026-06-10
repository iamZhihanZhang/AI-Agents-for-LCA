# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Compute bounding boxes and dimensions for PCB

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import cv2
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist

import pytesseract
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from .yolo_pcb_segmentation import segment as segment_pcb


# ========================= Math =========================
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def crop(image, p1, p2):
    return image[p1[1] : p2[1], p1[0] : p2[0]].copy()


def draw_rect(image, p1, p2, color):
    image[p1[1] : p2[1], p1[0] : p2[0]] = color


def crop_and_rotate_horizontal(image, p1, p2):
    cropped = crop(image, p1, p2)
    height, width = cropped.shape[:2]
    if height > width:
        cropped = imutils.rotate_bound(cropped, 90)
    return cropped


def shrink_bounding_box(p1, p2, image):
    # use edge detection to find the smallest bounding box inside the given bounding box
    cropped = image[p1[1] : p2[1], p1[0] : p2[0]].copy()
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)  # type: ignore
    edged = cv2.erode(edged, None, iterations=1)  # type: ignore

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    boxes = [cv2.minAreaRect(c) for c in cnts]
    areas = [w * h for _, (w, h), _ in boxes]

    if len(areas) == 0:
        rotated_rect_from_p1_p2 = cv2.minAreaRect(
            np.array([[p1, (p1[0], p2[1]), p2, (p2[0], p1[1])]])
        )
        return rotated_rect_from_p1_p2, cropped

    # box with the largest area is the target
    top_box = boxes[np.argmax(areas)]
    # add the offset from the crop
    top_box = ((top_box[0][0] + p1[0], top_box[0][1] + p1[1]), top_box[1], top_box[2])

    return top_box, cropped


# ========================= Segment Anything Bounding Boxes =========================
sam = sam_model_registry["vit_b"](
    checkpoint=os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".models", "sam_vit_b.pth"
    )
)
mask_generator = SamAutomaticMaskGenerator(sam)


def generate_sam_masks(image):
    masks = mask_generator.generate(image)
    images = []
    for mask in masks:
        img = image.copy()
        width, height = int(mask["bbox"][2]), int(mask["bbox"][3])
        x1, y1 = int(mask["bbox"][0]), int(mask["bbox"][1])
        x2, y2 = x1 + width, y1 + height
        segmentation = mask["segmentation"][y1:y2, x1:x2]
        segment = image[y1:y2, x1:x2][segmentation]
        if segment.size < 3000:  # discard small segments
            continue
        img[y1:y2, x1:x2][segmentation] = [0, 0, 255]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        images.append(img)
    return masks, images


# ========================= Edge Detection Bounding Boxes =========================
def generate_edge_boxes(image):
    copy = image.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)  # type: ignore
    edged = cv2.erode(edged, None, iterations=1)  # type: ignore

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    contour_boxes = []
    images = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        img = image.copy()
        images.append(img)

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # type: ignore
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)

        # find min and max x and y coordinates
        x1, y1 = box.min(axis=0).astype(int)
        x2, y2 = box.max(axis=0).astype(int)

        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)

        contour_boxes.append((x1, y1, x2, y2))

    return contour_boxes, images


# ========================= OCR Bounding Boxes =========================
def generate_character_boxes(image):
    character_boxes = []
    copy = image.copy()
    for img, rotation in [
        (image.copy(), 0),
        (imutils.rotate_bound(image, 90), 90),
        # (imutils.rotate_bound(image, -90), -90),
        # (imutils.rotate_bound(image, 180), 180),
    ]:  # check all orientations since the rulers can be vertical or horizontal
        height, width = img.shape[:2]
        boxes = pytesseract.image_to_boxes(
            img,
            output_type=pytesseract.Output.DICT,
            config='-c tessedit_char_whitelist="0123456789 " --psm 4',
        )
        for i in range(len(boxes["char"])):

            x1, y1, x2, y2 = (
                boxes["right"][i],
                height - boxes["top"][i],
                boxes["left"][i],
                height - boxes["bottom"][i],
            )

            # discard big boxes since they are not likely to be numbers from the rulers
            if abs((x2 - x1) * (y2 - y1)) > 1000:
                continue

            if rotation == 90:
                x1, y1, x2, y2 = y2, width - x1, y1, width - x2
            elif rotation == -90:
                x1, y1, x2, y2 = height - y1, x2, height - y2, x1
            cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                copy,
                boxes["char"][i],
                (x2, y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            character_boxes.append((x1, y1, x2, y2))

    return character_boxes, copy


# ========================= Select Candidates =========================
def select_pcb_by_character_boxes(image, masks, contour_boxes, character_boxes):
    # identify the pcb based on the number of characters inside the area
    pcb_mask = None
    pcb_mask_character_count = np.inf
    pcb_mask_size = -1
    for mask in masks:
        img = image.copy()

        width, height = int(mask["bbox"][2]), int(mask["bbox"][3])
        x1, y1 = int(mask["bbox"][0]), int(mask["bbox"][1])
        x2, y2 = x1 + width, y1 + height

        segmentation = mask["segmentation"][y1:y2, x1:x2]
        segment = image[y1:y2, x1:x2][segmentation]

        # ignore if average color is not dark or segment is too small
        if segment.size < 3000 or segment.mean() > 100:
            continue

        img[y1:y2, x1:x2][segmentation] = [0, 0, 255]

        # count number of character boxes inside the mask
        count = 0
        for box in character_boxes:
            if box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2:
                count += 1

        if count < pcb_mask_character_count or (
            count == pcb_mask_character_count and segment.size > pcb_mask_size
        ):
            pcb_mask_character_count = count
            pcb_mask = (x1, y1), (x2, y2), img
            pcb_mask_size = segment.size

    for x1, y1, x2, y2 in contour_boxes:
        img = image.copy()

        # ignore if average color is not dark
        segment = image[y1:y2, x1:x2]
        if segment.size < 3000 or segment.mean() > 165:
            continue
        img[y1:y2, x1:x2] = [255, 0, 0]

        # count number of character boxes inside the box
        count = 0
        for box in character_boxes:
            if box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2:
                count += 1

        if count < pcb_mask_character_count or (
            count == pcb_mask_character_count and segment.size > pcb_mask_size
        ):
            pcb_mask_character_count = count
            pcb_mask = (x1, y1), (x2, y2), img
            pcb_mask_size = segment.size

    p1, p2, img = pcb_mask  # type: ignore

    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    return p1, p2, img


def select_pcb_with_most_components(image, masks, contour_boxes):
    components = segment_pcb(image)[
        "predictions"
    ]  # [{ "x": int, "y": int, "width": int, "height": int, "confidence": float, "class": str }]

    # identify the pcb as the one with the most components
    pcb_mask = None
    pcb_mask_component_count = -np.inf
    pcb_mask_size = -1
    for mask in masks:
        img = image.copy()

        width, height = int(mask["bbox"][2]), int(mask["bbox"][3])
        x1, y1 = int(mask["bbox"][0]), int(mask["bbox"][1])
        x2, y2 = x1 + width, y1 + height

        segmentation = mask["segmentation"][y1:y2, x1:x2]
        segment = image[y1:y2, x1:x2][segmentation]

        # ignore if average color is not dark or segment is too small
        if segment.size < 3000 or segment.mean() > 100:
            continue

        img[y1:y2, x1:x2][segmentation] = [0, 0, 255]

        # count number of components inside the mask
        count = 0
        for component in components:
            if (
                component["x"] >= x1
                and component["y"] >= y1
                and component["x"] + component["width"] <= x2
                and component["y"] + component["height"] <= y2
            ):
                count += 1

        if count > pcb_mask_component_count or (
            count == pcb_mask_component_count and segment.size > pcb_mask_size
        ):
            pcb_mask_component_count = count
            pcb_mask = (x1, y1), (x2, y2), img
            pcb_mask_size = segment.size

    for x1, y1, x2, y2 in contour_boxes:
        img = image.copy()

        # ignore if average color is not dark
        segment = image[y1:y2, x1:x2]
        if segment.size < 3000 or segment.mean() > 165:
            continue
        img[y1:y2, x1:x2] = [255, 0, 0]

        # count number of components inside the box
        count = 0
        for component in components:
            if (
                component["x"] >= x1
                and component["y"] >= y1
                and component["x"] + component["width"] <= x2
                and component["y"] + component["height"] <= y2
            ):
                count += 1

        if count > pcb_mask_component_count or (
            count == pcb_mask_component_count and segment.size > pcb_mask_size
        ):
            pcb_mask_component_count = count
            pcb_mask = (x1, y1), (x2, y2), img
            pcb_mask_size = segment.size

    p1, p2, img = pcb_mask  # type: ignore

    cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

    return p1, p2, img


def find_ruler_candidates(image, masks, contour_boxes, character_boxes):
    candidate_rulers = []
    for mask in masks:
        img = image.copy()

        width, height = int(mask["bbox"][2]), int(mask["bbox"][3])
        x1, y1 = int(mask["bbox"][0]), int(mask["bbox"][1])
        x2, y2 = x1 + width, y1 + height

        segmentation = mask["segmentation"][y1:y2, x1:x2]
        segment = image[y1:y2, x1:x2][segmentation]

        # ignore if not big enough
        if segment.size < 3000:
            continue

        img[y1:y2, x1:x2][segmentation] = [0, 0, 255]

        candidate_rulers.append(((x1, y1, x2, y2), img))

    for x1, y1, x2, y2 in contour_boxes:
        img = image.copy()

        # ignore if not big enough
        segment = image[y1:y2, x1:x2]
        if segment.size < 3000:
            continue

        img[y1:y2, x1:x2] = [255, 0, 0]

        candidate_rulers.append(((x1, y1, x2, y2), img))

    return candidate_rulers


def select_best_ruler_candidate(image, candidate_rulers, top_box, character_boxes):
    ruler = None
    ruler_character_count = -1
    ruler_aspect_ratio = 0
    for bounding_box, img in candidate_rulers:
        # must not be more than 20% of the image
        if (
            abs(
                (bounding_box[2] - bounding_box[0])
                * (bounding_box[3] - bounding_box[1])
            )
            > 0.2 * image.size
        ):
            continue

        # must not entirely contain another candidate ruler
        # if any([bounding_box[0] <= box[0][0] and bounding_box[1] <= box[0][1] and bounding_box[2] >= box[0][2] and bounding_box[3] >= box[0][3] for box in candidate_rulers if box != (bounding_box, img)]):
        #     continue

        # must not contain the top_box which is the pcb
        if (
            bounding_box[0] <= top_box[0][0]
            and bounding_box[1] <= top_box[0][1]
            and bounding_box[2] >= top_box[0][0]
            and bounding_box[3] >= top_box[0][1]
        ):
            continue

        # count number of character boxes inside the mask
        count = 0
        for box in character_boxes:
            if (
                box[0] >= bounding_box[0]
                and box[1] >= bounding_box[1]
                and box[2] <= bounding_box[2]
                and box[3] <= bounding_box[3]
            ):
                count += 1

        # tie break with aspect ratio
        aspect_ratio = (bounding_box[2] - bounding_box[0]) / (
            bounding_box[3] - bounding_box[1]
        )

        if count > ruler_character_count:
            ruler_character_count = count
            ruler = (
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                img,
            )
            ruler_aspect_ratio = aspect_ratio
        elif count == ruler_character_count and abs(1 - aspect_ratio) > abs(
            1 - ruler_aspect_ratio
        ):
            ruler = (
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                img,
            )
            ruler_aspect_ratio = aspect_ratio

    p1, p2, img = ruler  # type: ignore

    return p1, p2, img


# ========================= Compute Dimensions =========================
def read_ruler_with_ocr(image, cropped):
    # numbers and spaces only
    thresh = cv2.adaptiveThreshold(
        cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    text = pytesseract.image_to_string(
        thresh, config='-c tessedit_char_whitelist="0123456789 " --psm 6'
    )
    text += " " + pytesseract.image_to_string(
        cropped, config='-c tessedit_char_whitelist="0123456789 " --psm 6'
    )
    # find the largest number in the text
    numbers = [int(s) for s in text.split() if s.isdigit()]
    if len(numbers) == 0:
        thresh = cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        text = pytesseract.image_to_string(
            thresh, config='-c tessedit_char_whitelist="0123456789 " --psm 6'
        )
        text += " " + pytesseract.image_to_string(
            image, config='-c tessedit_char_whitelist="0123456789 " --psm 6'
        )
        numbers = [int(s) for s in text.split() if s.isdigit()]
    # discard numbers larger than 500
    numbers = [n for n in numbers if n < 500]
    largest_number = max(numbers)

    # convert to inches
    if largest_number > 30:
        # assume mm
        largest_number /= 25.4
    elif largest_number >= 7:
        # assume cm
        largest_number /= 2.54

    # account for extra space on either end of ruler
    if cropped.mean() < 100:
        largest_number *= 1.5  # hard to read numbers are likely to be underestimated
    else:
        largest_number *= 1.3

    # pixel width to largest number ratio
    pixelsPerMetric = cropped.shape[1] / largest_number
    return pixelsPerMetric


def calc_dimensions(image, top_box, pixelsPerMetric):
    output = image.copy()

    box = cv2.cv.BoxPoints(top_box) if imutils.is_cv2() else cv2.boxPoints(top_box)  # type: ignore
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(output, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for x, y in box:
        cv2.circle(output, (int(x), int(y)), 5, (0, 0, 255), -1)
    tl, tr, br, bl = box
    tltrX, tltrY = midpoint(tl, tr)
    blbrX, blbrY = midpoint(bl, br)
    tlblX, tlblY = midpoint(tl, bl)
    trbrX, trbrY = midpoint(tr, br)
    # draw the midpoints on the image
    cv2.circle(output, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(output, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(output, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(output, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(
        output, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2
    )
    cv2.line(
        output, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2
    )

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
    cv2.putText(
        output,
        "{:.1f}in".format(dimB),
        (int(tltrX - 15), int(tltrY - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (55, 55, 55),
        2,
    )
    cv2.putText(
        output,
        "{:.1f}in".format(dimA),
        (int(trbrX + 10), int(trbrY)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (55, 55, 55),
        2,
    )

    return dimA, dimB, output


if __name__ == "__main__":
    # example/test usage
    from .yolo_pcb_detection import detect_pcb_axis_aligned_bounding_boxes

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    internal_photo_with_pcb = os.path.join(data_dir, "iPhone14ProMaxInternal.jpeg")
    image = cv2.imread(internal_photo_with_pcb)

    character_boxes, character_image = generate_character_boxes(image)
    contour_boxes, countour_images = generate_edge_boxes(image)
    # SAM is slow, can be disabled for faster inference at the expense of some accuracy loss
    masks = []
    # masks, mask_images = generate_sam_masks(image)

    detected_pcb = detect_pcb_axis_aligned_bounding_boxes(internal_photo_with_pcb)
    if detected_pcb is not None and len(detected_pcb.xyxy) > 0:
        pcb_box = detected_pcb.xyxy[0]
        pcb_box = (int(pcb_box[0]), int(pcb_box[1])), (int(pcb_box[2]), int(pcb_box[3]))
        just_pcb_image = crop(image, pcb_box[0], pcb_box[1])
        image_with_pcb_highlighted = image.copy()
        draw_rect(image_with_pcb_highlighted, pcb_box[0], pcb_box[1], (255, 255, 0))
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

    pixelsPerInch = read_ruler_with_ocr(image, just_ruler_image)
    pixelsPerMM = pixelsPerInch / 25.4
    pcb_width, pcb_height = (
        just_pcb_image.shape[1] / pixelsPerMM,
        just_pcb_image.shape[0] / pixelsPerMM,
    )
    print(f"Estimated PCB size: {pcb_width:.2f}x{pcb_height:.2f}mm")

    components = segment_pcb(just_pcb_image)
    for pred in components["predictions"]:
        w, h = (
            int(pred["width"]),
            int(pred["height"]),
        )
        confidence, category = pred["confidence"], pred["class"]
        print(
            f"{category},{confidence:.2f},{w / pixelsPerMM:.2f},{h / pixelsPerMM:.2f}"
        )

    cv2.imshow("image_with_pcb_highlighted", image_with_pcb_highlighted)
    cv2.imshow("ruler_image", just_ruler_image)
    cv2.imshow("components_annotated", components["annotated_image"])
    cv2.waitKey()

    cv2.destroyAllWindows()
