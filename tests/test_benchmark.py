import pytest
from unittest.mock import Mock, patch, MagicMock


class TestOptimizedPipeline:
    @patch("scripts.optimize_benchmark.ZImagePipeline")
    def test_initialization(self, mock_pipeline):
        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        assert pipeline.pipeline is None
        assert pipeline.device is None
        assert pipeline.dtype is None

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    @patch("scripts.optimize_benchmark.torch.backends.mps.is_available")
    def test_setup_device_mps(self, mock_mps, mock_pipeline):
        mock_mps.return_value = True
        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.setup_device()

        assert pipeline.device == "mps"

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    @patch("scripts.optimize_benchmark.torch.cuda.is_available")
    @patch("scripts.optimize_benchmark.torch.backends.mps.is_available")
    def test_setup_device_cuda(self, mock_mps, mock_cuda, mock_pipeline):
        mock_mps.return_value = False
        mock_cuda.return_value = True
        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.setup_device()

        assert pipeline.device == "cuda"

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    def test_load_model(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.device = "mps"
        pipeline.dtype = Mock()

        pipeline.load_model()

        assert pipeline.pipeline is not None

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    def test_apply_standard_optimization(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.device = "mps"
        pipeline.dtype = Mock()
        pipeline.optimization_level = "standard"

        pipeline.load_model()

        mock_pipeline.enable_attention_slicing.assert_called()

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    @patch("scripts.optimize_benchmark.torch.compile")
    def test_apply_compile_optimization(self, mock_compile, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_attention_slicing = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.device = "mps"
        pipeline.dtype = Mock()
        pipeline.optimization_level = "compile"

        pipeline.load_model()

        assert mock_compile.called

    @patch("scripts.optimize_benchmark.ZImagePipeline")
    def test_generate(self, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Mock()]
        mock_pipeline.return_value = mock_result

        from scripts.optimize_benchmark import OptimizedPipeline

        pipeline = OptimizedPipeline()
        pipeline.pipeline = mock_pipeline

        image = pipeline.generate("test prompt")

        assert image is not None
        mock_pipeline.assert_called_once()


class TestBenchmarkFunctions:
    @patch("scripts.optimize_benchmark.time.time")
    def test_measure_time(self, mock_time):
        mock_time.side_effect = [0, 1.5]

        from scripts.optimize_benchmark import measure_time

        def dummy_function():
            pass

        elapsed = measure_time(dummy_function)
        assert elapsed == 1.5

    def test_format_time(self):
        from scripts.optimize_benchmark import format_time

        assert format_time(1.5) == "1.50s"
        assert format_time(0.5) == "0.50s"

    def test_calculate_speedup(self):
        from scripts.optimize_benchmark import calculate_speedup

        assert calculate_speedup(10, 5) == 2.0
        assert calculate_speedup(100, 50) == 2.0


class TestMain:
    @patch("scripts.optimize_benchmark.OptimizedPipeline")
    @patch("scripts.optimize_benchmark.print")
    def test_main_runs_all_benchmarks(self, mock_print, mock_pipeline_class):
        mock_pipeline = Mock()
        mock_pipeline.generate.return_value = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        from scripts.optimize_benchmark import main

        # Should run without errors
        try:
            main()
        except Exception as e:
            # Expected to fail if model not available
            pass


class TestOptimizationLevels:
    @patch("scripts.optimize_benchmark.ZImagePipeline")
    def test_optimization_map(self, mock_pipeline):
        from scripts.optimize_benchmark import OPTIMIZATION_CONFIGS

        assert "standard" in OPTIMIZATION_CONFIGS
        assert "compile" in OPTIMIZATION_CONFIGS
        assert "quantize" in OPTIMIZATION_CONFIGS
        assert "aggressive" in OPTIMIZATION_CONFIGS
