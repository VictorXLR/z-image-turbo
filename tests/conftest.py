import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image


@pytest.fixture
def sample_prompt():
    return "a lovely cat sitting on a windowsill, photorealistic, detailed, 4k"


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.to = Mock(return_value=pipeline)
    pipeline.enable_attention_slicing = Mock()
    return pipeline


@pytest.fixture
def mock_device():
    return "mps"


@pytest.fixture
def mock_dtype():
    import torch

    return torch.bfloat16
