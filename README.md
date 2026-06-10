# Sustainability Assessment using Multimodal AI Agents

[Zhihan Zhang](https://homes.cs.washington.edu/~zzhihan/),
[Alexander Metzger](https://www.linkedin.com/in/alexander-le-metzger/),
[Yuxuan Mei](https://www.wesleyan.edu/about/directory/profile.html?id=ymei),
[Felix Hähnlein](https://obikate.github.io/),
[Zachary Englhardt](https://zachary.englhardt.com/),
[Tingyu Cheng](https://tingyucheng.com),
[Gregory D. Abowd](https://coe.northeastern.edu/people/abowd-gregory/),
[Shwetak Patel](https://www.cs.washington.edu/people/faculty/shwetak-patel/),
[Adriana Schulz](https://www.computationaldesign.group/adriana),
[Vikram Iyer](https://homes.cs.washington.edu/~vsiyer/)


## Overview
Reducing the growing environmental impact of the computing industry requires assessing the emissions of electronics at scale. However, a traditional life-cycle assessment (LCA) of an electronic device, which maps materials and processes to environmental impacts, often requires proprietary or unavailable data. Here we report a multimodal multi-agent artificial intelligence system that emulates the collaborative process between LCA professionals and stakeholders (such as product managers and engineers) to estimate the carbon footprint of electronic devices. The agents iteratively construct a complete life-cycle inventory by leveraging a structured data abstraction and software tools that mine information from the public Internet, including repair communities and government regulatory databases. This reduces data gaps and data collection from weeks or months of expert time to under 1 min. The system can calculate the carbon footprint within 19% of expert LCAs with zero proprietary data (typical of the variation between human LCAs). We also show that by encoding domain-specific knowledge, environmental impact estimation can be reframed as a data-driven prediction task, in which both unknown products and emission factors are represented as weighted combinations of similar ones with known emissions.

![Sustainability Assessment using Multimodal AI Agents](/figures/overview.jpg)
---

For more information, read the full paper published in [Nature Electronics](https://www.nature.com/articles/s41928-026-01653-w).

## Getting Started
Following user feedback, we've recognized that the developed methodologies have broader applications in LCA. Therefore, we've separated the codebase to allow independent execution of different components.

### Repository Structure

```
.
├── agents_for_lifecycle_inventory/     # Multi-agent pipeline for life cycle inventory construction
└── estimation_using_textual_features/  # Weighted k-NN estimator for CO2e prediction
```

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Tesseract OCR (`setup_system.sh` installs this)
- API keys: OpenAI or Gemini, Google Search

### Installation

```bash
git clone https://github.com/iamZhihanZhang/AI-Agents-for-LCA
cd AI-Agents-for-LCA/agents_for_lifecycle_inventory

# Install system dependencies (Tesseract, etc.)
bash scripts/setup_system.sh

# Set up Python environment
bash scripts/setup_python.sh

# Download ML model weights (SAM, YOLO)
bash scripts/setup_models.sh

# Copy and fill in API keys
cp .env.example .env
```

### Running the Agent Pipeline

```bash
cd agents_for_lifecycle_inventory
python scripts/run_self_play.py
```

### Running the k-NN Estimator

```bash
cd estimation_using_textual_features
python run_experiments.py
```

---

## Citation

If you use this repository in academic work, please cite it:

```bibtex
@article{zhang_sustainability_2026,
title = {Sustainability assessment using multimodal artificial intelligence agents},
year = {2026},
url = {https://www.nature.com/articles/s41928-026-01653-w},
doi = {10.1038/s41928-026-01653-w},
journal = {Nature Electronics},
publisher = {Nature Publishing Group},
author = {Zhang, Zhihan and Metzger, Alexander and Mei, Yuxuan and Hähnlein, Felix and Englhardt, Zachary and Cheng, Tingyu and Abowd, Gregory D. and Patel, Shwetak and Schulz, Adriana and Iyer, Vikram},
}
```

## Contact

If you have any questions, feel free to contact us via email at zzhihan@cs.washington.edu or open a GitHub Issue.