"""Tests for CLI module."""

import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, MagicMock

from voice_to_text.cli import main


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


class TestTranscribeCommand:
    """Tests for transcribe command."""
<<<<<<< HEAD
    
    def test_help(self, runner):
        """Show help message."""
        result = runner.invoke(main, ["transcribe", "--help"])
        
        assert result.exit_code == 0
        assert "Transcribe voice messages" in result.output
    
=======

    def test_help(self, runner):
        """Show help message."""
        result = runner.invoke(main, ["transcribe", "--help"])

        assert result.exit_code == 0
        assert "Transcribe voice messages" in result.output

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_single_file(self, mock_create, runner, sample_wav):
        """Transcribe single file."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="Test transcription",
            language="en",
            duration=2.0,
            segments=[]
        )
        mock_create.return_value = mock_transcriber
<<<<<<< HEAD
        
        result = runner.invoke(main, ["transcribe", str(sample_wav)])
        
        assert result.exit_code == 0
        assert "Test transcription" in result.output
    
=======

        result = runner.invoke(main, ["transcribe", str(sample_wav)])

        assert result.exit_code == 0
        assert "Test transcription" in result.output

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_output_to_file(self, mock_create, runner, sample_wav, temp_dir):
        """Save transcription to file."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="Saved transcription",
            language="en",
            duration=2.0,
            segments=[]
        )
        mock_create.return_value = mock_transcriber
<<<<<<< HEAD
        
        output_file = temp_dir / "output.txt"
        
=======

        output_file = temp_dir / "output.txt"

>>>>>>> 243afb8 (init commit  in github)
        result = runner.invoke(main, [
            "transcribe", str(sample_wav),
            "-o", str(output_file)
        ])
<<<<<<< HEAD
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Saved transcription" in output_file.read_text()
    
=======

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Saved transcription" in output_file.read_text()

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_json_format(self, mock_create, runner, sample_wav):
        """Output in JSON format."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="JSON test",
            language="en",
            duration=2.0,
            segments=[]
        )
        mock_create.return_value = mock_transcriber
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        result = runner.invoke(main, [
            "transcribe", str(sample_wav),
            "-f", "json"
        ])
<<<<<<< HEAD
        
        assert result.exit_code == 0
        assert '"text"' in result.output
        assert '"language"' in result.output
    
=======

        assert result.exit_code == 0
        assert '"text"' in result.output
        assert '"language"' in result.output

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_specify_language(self, mock_create, runner, sample_wav):
        """Specify transcription language."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="Тест",
            language="ru",
            duration=2.0,
            segments=[]
        )
        mock_create.return_value = mock_transcriber
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        result = runner.invoke(main, [
            "transcribe", str(sample_wav),
            "-l", "ru"
        ])
<<<<<<< HEAD
        
        assert result.exit_code == 0
        mock_transcriber.transcribe.assert_called_once()
    
=======

        assert result.exit_code == 0
        mock_transcriber.transcribe.assert_called_once()
        call_args = mock_transcriber.transcribe.call_args
        assert call_args[1].get("language") == "ru" or call_args[0][1] == "ru"

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_model_selection(self, mock_create, runner, sample_wav):
        """Select model size."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = MagicMock(
            text="Test",
            language="en",
            duration=1.0,
            segments=[]
        )
        mock_create.return_value = mock_transcriber
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        result = runner.invoke(main, [
            "transcribe", str(sample_wav),
            "-m", "medium"
        ])
<<<<<<< HEAD
        
        assert result.exit_code == 0
        from voice_to_text.transcriber import ModelSize
        mock_create.assert_called_once()
        assert mock_create.call_args[1]["model_size"] == ModelSize.MEDIUM
    
=======

        assert result.exit_code == 0
        # Verify model_size was passed
        from voice_to_text.transcriber import ModelSize
        mock_create.assert_called_once()
        assert mock_create.call_args[1]["model_size"] == ModelSize.MEDIUM

>>>>>>> 243afb8 (init commit  in github)
    @patch('voice_to_text.cli.create_transcriber')
    def test_batch_processing(self, mock_create, runner, voice_files_dir):
        """Batch process directory."""
        mock_transcriber = MagicMock()
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        def mock_batch(files, language=None):
            for f in files:
                yield f, MagicMock(
                    text=f"Transcription of {f.name}",
                    language="en",
                    duration=1.0
                )
<<<<<<< HEAD
        
        mock_transcriber.transcribe_batch = mock_batch
        mock_create.return_value = mock_transcriber
        
=======

        mock_transcriber.transcribe_batch = mock_batch
        mock_create.return_value = mock_transcriber

>>>>>>> 243afb8 (init commit  in github)
        result = runner.invoke(main, [
            "transcribe", str(voice_files_dir),
            "--batch"
        ])
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        assert result.exit_code == 0
        assert "Summary" in result.output


class TestModelsCommand:
    """Tests for models command."""
<<<<<<< HEAD
    
    def test_list_models(self, runner):
        """List available models."""
        result = runner.invoke(main, ["models"])
        
=======

    def test_list_models(self, runner):
        """List available models."""
        result = runner.invoke(main, ["models"])

>>>>>>> 243afb8 (init commit  in github)
        assert result.exit_code == 0
        assert "tiny" in result.output
        assert "base" in result.output
        assert "small" in result.output
        assert "medium" in result.output
        assert "large" in result.output
        assert "Parameters" in result.output
        assert "VRAM" in result.output


class TestMainGroup:
    """Tests for main CLI group."""
<<<<<<< HEAD
    
    def test_version(self, runner):
        """Show version."""
        result = runner.invoke(main, ["--version"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    def test_no_command(self, runner):
        """No command shows help."""
        result = runner.invoke(main)
        
        assert result.exit_code == 0
        assert "Voice-to-Text" in result.output
=======

    def test_version(self, runner):
        """Show version."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_command(self, runner):
        """No command shows help."""
        result = runner.invoke(main)

        assert result.exit_code == 0
        assert "Voice-to-Text" in result.output
>>>>>>> 243afb8 (init commit  in github)
