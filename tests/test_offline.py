from unittest.mock import patch, MagicMock
import os

import pytest


@pytest.fixture
def mock_model_loading():
    """Mock all model loading calls to avoid downloading models."""
    with (
        patch("manga_ocr.ocr.ViTImageProcessor.from_pretrained") as mock_processor,
        patch("manga_ocr.ocr.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("manga_ocr.ocr.MangaOcrModel.from_pretrained") as mock_model,
        patch("manga_ocr.ocr.MangaOcr.__call__", return_value=""),
    ):
        mock_model.return_value = MagicMock()
        mock_model.return_value.device = "cpu"
        mock_model.return_value.cuda = MagicMock()

        yield mock_processor, mock_tokenizer, mock_model


def test_force_offline_passes_local_files_only(mock_model_loading):
    from manga_ocr import MangaOcr

    mock_processor, mock_tokenizer, mock_model = mock_model_loading
    MangaOcr(force_offline=True, force_cpu=True)

    mock_processor.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)
    mock_tokenizer.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)
    mock_model.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)


def test_default_does_not_force_local_files_only(mock_model_loading):
    from manga_ocr import MangaOcr

    mock_processor, mock_tokenizer, mock_model = mock_model_loading
    MangaOcr(force_cpu=True)

    mock_processor.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=False)
    mock_tokenizer.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=False)
    mock_model.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=False)


def test_transformers_offline_env_var(mock_model_loading):
    from manga_ocr import MangaOcr

    mock_processor, mock_tokenizer, mock_model = mock_model_loading
    with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "1"}):
        MangaOcr(force_cpu=True)

    mock_processor.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)
    mock_tokenizer.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)
    mock_model.assert_called_once_with("kha-white/manga-ocr-base", local_files_only=True)
