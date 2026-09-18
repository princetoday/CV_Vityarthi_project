import cv2
import numpy as np

from src.analyzer import analyze_image
from src.scoring import classify_quality


def test_quality_classification():
    assert classify_quality(90) == "Excellent"
    assert classify_quality(70) == "Good"
    assert classify_quality(50) == "Fair"
    assert classify_quality(20) == "Poor"


def test_analyze_image(tmp_path):
    image = np.full(
        (200, 200, 3),
        128,
        dtype=np.uint8
    )

    cv2.rectangle(
        image,
        (50, 50),
        (150, 150),
        (255, 255, 255),
        -1
    )

    image_path = tmp_path / "test.png"

    cv2.imwrite(
        str(image_path),
        image
    )

    result = analyze_image(
        str(image_path)
    )

    assert "metrics" in result
    assert "scores" in result
    assert "label" in result

    assert 0 <= result["scores"]["overall"] <= 100
