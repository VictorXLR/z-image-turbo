import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestGenerateScript:
    @patch("src.generate.ZImagePipeline")
    @patch("src.generate.torch.backends.mps.is_available")
    def test_setup_device_mps(self, mock_mps, mock_pipeline):
        mock_mps.return_value = True
        from src.generate import setup_device

        device, dtype = setup_device()
        assert device == "mps"

    @patch("src.generate.ZImagePipeline")
    @patch("src.generate.torch.backends.mps.is_available")
    @patch("src.generate.torch.cuda.is_available")
    def test_setup_device_cuda(self, mock_cuda, mock_mps, mock_pipeline):
        mock_mps.return_value = False
        mock_cuda.return_value = True
        from src.generate import setup_device

        device, dtype = setup_device()
        assert device == "cuda"

    @patch("src.generate.ZImagePipeline")
    @patch("src.generate.torch.backends.mps.is_available")
    @patch("src.generate.torch.cuda.is_available")
    def test_setup_device_cpu(self, mock_cuda, mock_mps, mock_pipeline):
        mock_mps.return_value = False
        mock_cuda.return_value = False
        from src.generate import setup_device
        import torch

        device, dtype = setup_device()
        assert device == "cpu"
        assert dtype == torch.float32

    @patch("src.generate.ZImagePipeline")
    def test_load_pipeline_success(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from src.generate import load_pipeline

        pipeline = load_pipeline("mps", Mock())
        assert pipeline is not None
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("src.generate.ZImagePipeline")
    def test_load_pipeline_with_attention_slicing(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from src.generate import load_pipeline

        load_pipeline("cuda", Mock())
        mock_pipeline.enable_attention_slicing.assert_called_once()

    @patch("src.generate.ZImagePipeline")
    def test_load_pipeline_cpu_no_attention_slicing(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from src.generate import load_pipeline

        load_pipeline("cpu", Mock())
        mock_pipeline.enable_attention_slicing.assert_not_called()

    def test_parse_arguments_default(self):
        from src.generate import parse_args

        args = parse_args([])
        assert (
            args.prompt
            == "a lovely cat sitting on a windowsill, photorealistic, detailed, 4k"
        )
        assert args.width == 1024
        assert args.height == 1024
        assert args.steps == 9
        assert args.seed == -1

    def test_parse_arguments_custom(self):
        from src.generate import parse_args
        import shlex

        args = parse_args(
            shlex.split(
                "--prompt custom test --width 512 --height 512 --steps 6 --seed 42"
            )
        )
        assert args.prompt == "custom test"
        assert args.width == 512
        assert args.height == 512
        assert args.steps == 6
        assert args.seed == 42

    @patch("src.generate.torch.Generator")
    def test_main_with_valid_inputs(self, mock_generator):
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result

        from src.generate import main

        with patch("src.generate.load_pipeline", return_value=mock_pipeline):
            with patch("src.generate.setup_device", return_value=("mps", Mock())):
                with patch(
                    "src.generate.parse_args",
                    return_value=Mock(
                        prompt="test",
                        width=512,
                        height=512,
                        steps=6,
                        seed=42,
                        output="test.png",
                    ),
                ):
                    main()
                    mock_pipeline.assert_called_once()
                    mock_image.save.assert_called_once()

    @patch("src.generate.torch.Generator")
    def test_main_random_seed(self, mock_generator):
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_image = Mock(spec=Image.Image)
        mock_result.images = [mock_image]
        mock_pipeline.return_value = mock_result

        from src.generate import main

        with patch("src.generate.load_pipeline", return_value=mock_pipeline):
            with patch("src.generate.setup_device", return_value=("mps", Mock())):
                with patch(
                    "src.generate.parse_args",
                    return_value=Mock(
                        prompt="test",
                        width=512,
                        height=512,
                        steps=6,
                        seed=-1,
                        output="test.png",
                    ),
                ):
                    main()
                    mock_generator.assert_not_called()

    @patch("src.generate.ZImagePipeline")
    def test_load_pipeline_model_not_found(self, mock_pipeline_class):
        from src.generate import load_pipeline
        import torch

        mock_pipeline_class.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(Exception):
            load_pipeline("cpu", torch.float32)


class TestGenerateEdgeCases:
    def test_generator_creation_with_seed(self):
        from src.generate import torch

        generator = torch.Generator(device="cpu").manual_seed(42)
        assert generator is not None

    def test_model_id_constant(self):
        from src.generate import MODEL_ID

        assert MODEL_ID == "Tongyi-MAI/Z-Image-Turbo"
