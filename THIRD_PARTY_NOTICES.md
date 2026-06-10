# Third-Party Code and Data
We build on a couple open-source resources with permissive licenses as well as various proprietary and open-source LLM apis. The LLM APIs can be found [here](agents_for_lifecycle_inventory/agentic_lci/llm_adapters). The rest is documented below.

## Boavizta Environmental Footprint Data
Portions of the report parsing logic and datasets are derived from:

Boavizta - Environmental Footprint Data: [repo link](https://github.com/Boavizta/environmental-footprint-data)

License: [Open Database License ](http://opendatacommons.org/licenses/odbl/1.0/)

The original project provides parsers and datasets for extracting
environmental footprint information from manufacturer disclosures.
This repository adapts and extends that work for use in an agentic,
multi-modal LCA system.

## Roboflow PCB Component Detection
We use a pre-trained YOLOv5 from Roboflow to detect bounding boxes for PCB components: [model link](https://universe.roboflow.com/roboflow-100/printed-circuit-board/model/3)

This is part of the [Roboflow-100 project](https://universe.roboflow.com/roboflow-100) 

Paper: [arXiv](https://arxiv.org/abs/2211.13523)

License: [MIT](https://github.com/roboflow/roboflow-100-benchmark?tab=MIT-1-ov-file)

## Segment Anything
We use Segment Anything by Facebook Research for segmentation masks: [repo link](https://github.com/facebookresearch/segment-anything)

License: [Apache-2.0](https://github.com/facebookresearch/segment-anything?tab=Apache-2.0-1-ov-file#readme)

## Tesseract
We use Tesseract by Google for Optical Character Recognition: [repo link](https://github.com/tesseract-ocr/tesseract)

License: [Apache-2.0](https://github.com/tesseract-ocr/tesseract?tab=Apache-2.0-1-ov-file)
