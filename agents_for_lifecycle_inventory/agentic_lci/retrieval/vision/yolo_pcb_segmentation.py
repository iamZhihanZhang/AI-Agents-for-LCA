# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Segment PCB components using https://universe.roboflow.com/roboflow-100/printed-circuit-board/model/3

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import supervision as sv
from inference import get_model

from dotenv import load_dotenv

load_dotenv()

assert "ROBOFLOW_API_KEY" in os.environ, "Please add roboflow api key to .env file"

# load a pre-trained yolov8n model
model = get_model(model_id="printed-circuit-board/3")

CLASSES = [
    "Button",
    "Capacitor",
    "Capacitor Jumper",
    "Clock",
    "Connector",
    "Diode",
    "EM",
    "Electrolytic Capacitor",
    "Ferrite Bead",
    "IC",
    "Inductor",
    "Jumper",
    "Led",
    "Pads",
    "Pins",
    "Resistor",
    "Resistor Jumper",
    "Resistor Network",
    "Switch",
    "Test Point",
    "Transistor",
    "Unknown Unlabeled",
    "iC",
]


def divide_image_into_patches(image, patch_size=640):
    width, height = image.size
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, i + patch_size, j + patch_size)
            patch = image.crop(box)
            patches.append((patch, (i, j)))
    return patches


def segment(image):
    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image, confidence_threshold=0.3, iou_threshold=0.5)[0]

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)
    # {'xyxy': array([[ 273.,  843.,  305.,  943.],
    #         [ 275.,  852.,  306.,  949.],
    #         [ 276.,  632.,  870., 1635.]]),
    # 'mask': None,
    # 'confidence': array([0.72412109, 0.68457031, 0.58544922]),
    # 'class_id': array([9, 4, 9]),
    # 'tracker_id': None,
    # 'data': {'class_name': array(['IC', 'Connector', 'IC'], dtype='<U9')}}

    # format predictions
    predictions = []
    for xyxy, confidence, cls in zip(
        detections.xyxy,
        detections.confidence,  # type: ignore
        detections.data["class_name"],
    ):
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        pred = {
            "x": xyxy[0] + width / 2,
            "y": xyxy[1] + height / 2,
            "width": width,
            "height": height,
            "confidence": confidence,
            "class": cls,
        }
        predictions.append(pred)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image.copy(), detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    return {"predictions": predictions, "annotated_image": annotated_image}


if __name__ == "__main__":
    # example/test usage
    from PIL import Image

    results = segment(
        Image.open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "data",
                "iPhone14ProMaxInternal.jpeg",
            )
        )
    )
    print(results["predictions"])
    sv.plot_image(results["annotated_image"])
