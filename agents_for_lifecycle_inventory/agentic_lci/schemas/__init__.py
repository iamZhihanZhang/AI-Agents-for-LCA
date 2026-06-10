from .delta_lca import DeltaLCAInput, PROMPT_DESCRIPTION as DELTA_LCA_PROMPT_DESCRIPTION
from .electronics_gwp import (
    ProductReportModel,
    PROMPT_DESCRIPTION as ELECTRONICS_GWP_PROMPT_DESCRIPTION,
)
from .electronics import (
    ElectronicProduct,
    PROMPT_DESCRIPTION as ELECTRONICS_PROMPT_DESCRIPTION,
)
from .living_sustainability import (
    LCAInput,
    PROMPT_DESCRIPTION as LIVING_SUSTAINABILITY_PROMPT_DESCRIPTION,
)

SCHEMAS = [
    {
        "model": DeltaLCAInput,
        "use": "electronics components",
        "description": DELTA_LCA_PROMPT_DESCRIPTION,
    },
    {
        "model": ProductReportModel,
        "use": "electronics components",
        "description": ELECTRONICS_GWP_PROMPT_DESCRIPTION,
    },
    {
        "model": ElectronicProduct,
        "use": "electronics components",
        "description": ELECTRONICS_PROMPT_DESCRIPTION,
    },
    {
        "model": LCAInput,
        "use": "electronics components",
        "description": LIVING_SUSTAINABILITY_PROMPT_DESCRIPTION,
    },
]

SCHEMA_SUMMARIES = "\n\n".join(
    f'({i}) {s["model"].__name__}: {s["use"]}' for i, s in enumerate(SCHEMAS)
)
