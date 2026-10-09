"""Image preprocessing utilities.

The saved processed dataset intentionally uses a direct resize to a fixed square
input size. The raw dataset dimensions are strongly class-correlated, so padding
would create visible borders that a model could exploit as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps


ResizePolicy = Literal["direct_resize", "letterbox"]


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    mode: str
    format: str | None
    is_jpeg_header: bool


def has_jpeg_header(path: Path) -> bool:
    """Return True when a file starts with the JPEG magic bytes."""
    with path.open("rb") as file:
        return file.read(2) == b"\xff\xd8"


def read_image_metadata(path: Path) -> ImageMetadata:
    """Read image metadata and verify the file can be decoded."""
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        return ImageMetadata(
            width=image.width,
            height=image.height,
            mode=image.mode,
            format=image.format,
            is_jpeg_header=has_jpeg_header(path),
        )


def open_rgb_image(path: Path) -> Image.Image:
    """Open an image, apply EXIF orientation, and convert it to RGB."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
    policy: ResizePolicy = "direct_resize",
) -> Image.Image:
    """Resize an image with the selected project policy."""
    if policy == "direct_resize":
        return image.resize(size, Image.Resampling.LANCZOS)

    if policy == "letterbox":
        resized = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (0, 0, 0))
        left = (size[0] - resized.width) // 2
        top = (size[1] - resized.height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    raise ValueError(f"Unsupported resize policy: {policy}")


def save_processed_image(
    source_path: Path,
    destination_path: Path,
    image_size: tuple[int, int],
    resize_policy: ResizePolicy,
    jpeg_quality: int,
) -> None:
    """Create one clean RGB JPEG image for downstream model training."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with open_rgb_image(source_path) as image:
        processed = resize_image(image, image_size, resize_policy)
        processed.save(
            destination_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )
