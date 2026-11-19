"""Tiny helper so every chart is saved as PNG into the current folder."""
SAVED = []


def save(fig, name: str, dpi: int = 110):
    fig.savefig(name, dpi=dpi, bbox_inches="tight")
    if name not in SAVED:
        SAVED.append(name)
    return name
