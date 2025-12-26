import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestTerminalImageViewer:
    @pytest.fixture
    def viewer(self, temp_output_dir):
        with patch("src.cli.TerminalImageViewer.OUTPUT_DIR", temp_output_dir):
            from src.cli import TerminalImageViewer

            return TerminalImageViewer(width=640, height=480, steps=9)

    def test_format_bytes(self):
        from src.cli import format_bytes

        assert format_bytes(100) == "100B"
        assert format_bytes(2048) == "2KB"
        assert format_bytes(1048576) == "1MB"
        assert format_bytes(1073741824) == "1.0GB"

    def test_memory_bar(self):
        from src.cli import memory_bar

        bar = memory_bar(used=500, total=1000, width=10)
        assert len(bar) == 12  # color codes + bar + reset
        assert "█" in bar
        assert "░" in bar

    def test_format_bytes_negative(self):
        from src.cli import format_bytes

        result = format_bytes(0)
        assert result == "0B"

    def test_memory_bar_edge_cases(self):
        from src.cli import memory_bar

        # Zero total
        bar = memory_bar(used=100, total=0, width=10)
        assert bar.count("░") == 10

        # Over 100% usage
        bar = memory_bar(used=2000, total=1000, width=10)
        assert bar.count("█") == 10
        assert bar.count("░") == 0

    def test_save_image_basic(self, viewer, mock_image):
        test_prompt = "test cat"
        filepath = viewer.save_image(mock_image, test_prompt)

        assert filepath.exists()
        assert "test_cat" in filepath.name
        assert filepath.suffix == ".png"

    def test_save_image_unicode(self, viewer, mock_image):
        test_prompt = "猫咪"  # Chinese characters
        filepath = viewer.save_image(mock_image, test_prompt)

        assert filepath.exists()
        assert filepath.suffix == ".png"

    def test_save_image_special_chars(self, viewer, mock_image):
        test_prompt = "cat!@#$%^&*()"
        filepath = viewer.save_image(mock_image, test_prompt)

        assert filepath.exists()
        assert "cat" in filepath.name
        assert filepath.suffix == ".png"

    def test_save_image_empty_after_sanitization(self, viewer, mock_image):
        test_prompt = "!@#$%"
        filepath = viewer.save_image(mock_image, test_prompt)

        assert filepath.exists()
        assert "output" in filepath.name

    def test_save_image_incrementing_counter(self, viewer, mock_image):
        test_prompt = "increment test"

        # First file
        filepath1 = viewer.save_image(mock_image, test_prompt)
        assert filepath1.name == "increment_test_1.png"

        # Second file
        filepath2 = viewer.save_image(mock_image, test_prompt)
        assert filepath2.name == "increment_test_2.png"

    def test_save_image_long_prompt_truncated(self, viewer, mock_image):
        test_prompt = "a" * 50
        filepath = viewer.save_image(mock_image, test_prompt)

        assert filepath.exists()
        assert filepath.name.count("a") <= 30

    @patch("src.cli.subprocess.run")
    def test_copy_to_clipboard_success(self, mock_run, viewer, mock_image):
        mock_run.return_value = None
        viewer.copy_to_clipboard(mock_image)
        assert mock_run.called

    @patch("src.cli.subprocess.run")
    def test_copy_to_clipboard_failure(self, mock_run, viewer, mock_image):
        mock_run.side_effect = Exception("Test error")
        # Should not raise exception
        viewer.copy_to_clipboard(mock_image)

    def test_delete_image(self, viewer, mock_image, temp_output_dir):
        filepath = viewer.save_image(mock_image, "delete test")
        assert filepath.exists()

        viewer.current_filepath = filepath
        viewer.delete_image()

        assert not filepath.exists()
        assert viewer.current_filepath is None

    def test_delete_image_nonexistent(self, viewer, temp_output_dir):
        viewer.current_filepath = temp_output_dir / "nonexistent.png"
        # Should not raise exception
        viewer.delete_image()

    @patch("src.cli.torch.backends.mps.is_available")
    @patch("src.cli.ZImagePipeline")
    def test_setup_device_mps(self, mock_pipeline, mock_mps):
        mock_mps.return_value = True
        from src.cli import TerminalImageViewer

        viewer = TerminalImageViewer()
        viewer.setup_device()

        assert viewer.device == "mps"

    @patch("src.cli.torch.backends.mps.is_available")
    @patch("src.cli.torch.cuda.is_available")
    @patch("src.cli.ZImagePipeline")
    def test_setup_device_cuda(self, mock_pipeline, mock_cuda, mock_mps):
        mock_mps.return_value = False
        mock_cuda.return_value = True
        from src.cli import TerminalImageViewer

        viewer = TerminalImageViewer()
        viewer.setup_device()

        assert viewer.device == "cuda"

    @patch("src.cli.torch.backends.mps.is_available")
    @patch("src.cli.torch.cuda.is_available")
    @patch("src.cli.ZImagePipeline")
    def test_setup_device_cpu(self, mock_pipeline, mock_cuda, mock_mps):
        mock_mps.return_value = False
        mock_cuda.return_value = False
        from src.cli import TerminalImageViewer
        import torch

        viewer = TerminalImageViewer()
        viewer.setup_device()

        assert viewer.device == "cpu"
        assert viewer.dtype == torch.float32


class TestUtilityFunctions:
    def test_getch(self):
        from src.cli import getch

        # Mock stdin.read to return a character
        with patch("sys.stdin.fileno", return_value=1):
            with patch("sys.stdin.read", return_value="q"):
                result = getch()
                assert result == "q"

    def test_bell(self, capsys):
        from src.cli import bell

        bell()
        captured = capsys.readouterr()
        assert "\a" in captured.out

    def test_width_height_initialization(self):
        from src.cli import TerminalImageViewer

        viewer = TerminalImageViewer(width=800, height=600, steps=6)
        assert viewer.width == 800
        assert viewer.height == 600
        assert viewer.steps == 6
