# -*- coding: utf-8 -*-
"""
Dataset đã xử lý được lưu với kích thước vuông cố định bằng cách resize trực tiếp.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from PIL import Image, ImageOps

ResizePolicy = Literal["direct_resize", "letterbox"]

@dataclass(frozen=True)
class ImageMetadata:
    """
    Thông tin metadata cơ bản của một ảnh.
    """
    width: int
    height: int
    mode: str
    format: str | None
    is_jpeg_header: bool


def has_jpeg_header(path):
    """
    path: Path tới file ảnh.
    Trả về: True nếu 2 byte đầu là b"\\xff\\xd8" (JPEG), ngược lại False.
    """
    with path.open("rb") as file:
        header = file.read(2)
        return header == b"\xff\xd8"

def read_image_metadata(path):
    """
    path: Path tới file ảnh.
    Trả về: đối tượng ImageMetadata chứa width, height, mode, format, và
            thông tin có phải JPEG header hay không.
    """
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        metadata = ImageMetadata(
            width=image.width,
            height=image.height,
            mode=image.mode,
            format=image.format,
            is_jpeg_header=has_jpeg_header(path),
        )
        return metadata

def open_rgb_image(path):
    """
    path: Path tới file ảnh.
    Trả về: đối tượng PIL.Image ở chế độ RGB.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")

def resize_image(image, size, policy="direct_resize"):
    """
    image: đối tượng PIL.Image cần resize.
    size: tuple (width, height) kích thước đích.
    policy: "direct_resize" hoặc "letterbox".
    ảnh đã được resize 
    """
    if policy == "direct_resize":
        # Resize trực tiếp về kích thước đích, bỏ qua tỷ lệ gốc
        return image.resize(size, Image.Resampling.LANCZOS)
    if policy == "letterbox":
        # Giữ tỷ lệ, đặt vào giữa canvas đen (letterbox)
        resized = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (0, 0, 0))
        left = (size[0] - resized.width) // 2
        top = (size[1] - resized.height) // 2
        canvas.paste(resized, (left, top))
        return canvas
    raise ValueError("Chính sách resize không hỗ trợ: {}".format(policy))

def save_processed_image(
    source_path,
    destination_path,
    image_size,
    resize_policy,
    jpeg_quality,
):
    """
    source_path: Path tới ảnh thô.
    destination_path: Path tới file ảnh đã xử lý sẽ ghi ra.
    image_size: tuple (width, height) kích thước đích.
    resize_policy: chính sách resize ("direct_resize" hoặc "letterbox").
    jpeg_quality: chất lượng JPEG khi lưu (0–100).
    """
    if not destination_path.parent.exists():
        destination_path.parent.mkdir(parents=True)
    with open_rgb_image(source_path) as image:
        processed = resize_image(image, image_size, resize_policy)
        processed.save(
            destination_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
        )