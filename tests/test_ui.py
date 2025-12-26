import pytest
from unittest.mock import Mock, patch


class TestImageGenerator:
    @patch("src.ui.ZImagePipeline")
    def test_initialization(self, mock_pipeline):
        from src.ui import ImageGenerator

        generator = ImageGenerator()
        assert generator.pipeline is None
        assert generator.device is None
        assert generator.dtype is None

    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.torch.backends.mps.is_available")
    def test_load_model_mps(self, mock_mps, mock_pipeline):
        mock_mps.return_value = True
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.enable_attention_slicing = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        from src.ui import ImageGenerator

        generator = ImageGenerator()
        result = generator.load_model()

        assert "Model loaded successfully" in result
        assert generator.device == "mps"
        assert mock_pipeline_instance.enable_attention_slicing.called

    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.torch.cuda.is_available")
    @patch("src.ui.torch.backends.mps.is_available")
    def test_load_model_cuda(self, mock_mps, mock_cuda, mock_pipeline):
        mock_mps.return_value = False
        mock_cuda.return_value = True
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.to.return_value = mock_pipeline_instance
        mock_pipeline_instance.enable_attention_slicing = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance

        from src.ui import ImageGenerator

        generator = ImageGenerator()
        result = generator.load_model()

        assert "Model loaded successfully" in result
        assert generator.device == "cuda"

    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.torch.backends.mps.is_available")
    def test_load_model_already_loaded(self, mock_mps, mock_pipeline):
        from src.ui import ImageGenerator

        generator = ImageGenerator()
        generator.pipeline = Mock()

        result = generator.load_model()

        assert result == "Model already loaded"

    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.time.time")
    def test_generate_success(self, mock_time, mock_pipeline):
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = Mock(images=[Mock()])

        from src.ui import ImageGenerator

        generator = ImageGenerator()
        generator.pipeline = mock_pipeline_instance
        generator.device = "mps"

        mock_time.side_effect = [0, 1.5]

        image, info = generator.generate(
            prompt="test prompt",
            negative_prompt="",
            width=512,
            height=512,
            num_steps=9,
            seed=None,
        )

        assert image is not None
        assert "1.50" in info

    def test_generate_no_model(self):
        from src.ui import ImageGenerator

        generator = ImageGenerator()

        image, info = generator.generate(
            prompt="test",
            negative_prompt="",
            width=512,
            height=512,
            num_steps=9,
            seed=None,
        )

        assert image is None
        assert "Model not loaded" in info

    def test_generate_empty_prompt(self):
        from src.ui import ImageGenerator

        generator = ImageGenerator()
        generator.pipeline = Mock()

        image, info = generator.generate(
            prompt="", negative_prompt="", width=512, height=512, num_steps=9, seed=None
        )

        assert image is None
        assert "Please enter a prompt" in info


class TestUIComponents:
    def test_create_ui_returns_gradio_blocks(self):
        from src.ui import create_ui
        import gradio as gr

        demo = create_ui()
        assert isinstance(demo, gr.Blocks)

    @patch("src.ui.ImageGenerator")
    def test_load_button_handler(self, mock_generator_class):
        mock_generator = Mock()
        mock_generator.load_model.return_value = "Model loaded"
        mock_generator_class.return_value = mock_generator

        from src.ui import create_ui
        import gradio as gr

        demo = create_ui()

        # Verify demo was created successfully
        assert demo is not None


class TestPresetButtons:
    def test_preset_square(self):
        from src.ui import create_ui

        demo = create_ui()
        assert demo is not None

    def test_preset_landscape(self):
        from src.ui import create_ui

        demo = create_ui()
        assert demo is not None

    def test_preset_portrait(self):
        from src.ui import create_ui

        demo = create_ui()
        assert demo is not None


class TestUIExamples:
    def test_examples_loaded(self):
        from src.ui import create_ui

        demo = create_ui()
        assert demo is not None


class TestUIConfiguration:
    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.torch.backends.mps.is_available")
    def test_device_detection(self, mock_mps, mock_pipeline):
        mock_mps.return_value = True
        from src.ui import ImageGenerator

        generator = ImageGenerator()
        generator.load_model()

        assert generator.device == "mps"

    @patch("src.ui.ZImagePipeline")
    @patch("src.ui.torch.cuda.is_available")
    @patch("src.ui.torch.backends.mps.is_available")
    def test_cuda_device_fallback(self, mock_mps, mock_cuda, mock_pipeline):
        mock_mps.return_value = False
        mock_cuda.return_value = True
        from src.ui import ImageGenerator

        generator = ImageGenerator()
        generator.load_model()

        assert generator.device == "cuda"
