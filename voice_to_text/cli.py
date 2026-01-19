import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import json

from .transcriber import (
    create_transcriber,
    Backend,
    ModelSize,
    TranscriptionResult
)

console = Console()

# Поддерживаемые форматы голосовых
VOICE_EXTENSIONS = {".ogg", ".opus", ".m4a", ".mp3", ".wav", ".webm", ".amr"}


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Voice-to-Text: Bulk voice message transcription."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file (default: stdout)")
@click.option("-f", "--format", "output_format",
              type=click.Choice(["text", "json", "srt"]),
              default="text", help="Output format")
@click.option("-l", "--language", help="Language code (e.g., 'ru', 'en'). Auto-detect if not set")
@click.option("-m", "--model",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              default="base", help="Model size")
@click.option("-b", "--backend",
              type=click.Choice(["whisper", "faster", "api"]),
              default="faster", help="Transcription backend")
@click.option("--batch", is_flag=True, help="Process directory of files")
def transcribe(
        input_path: str,
        output: str | None,
        output_format: str,
        language: str | None,
        model: str,
        backend: str,
        batch: bool
):
    """Transcribe voice messages to text.

    Examples:

        # Single file
        vtt transcribe voice.ogg

        # Directory batch
        vtt transcribe ./voices --batch -o results.json -f json

        # Russian language, large model
        vtt transcribe voice.ogg -l ru -m large

        # Using OpenAI API
        vtt transcribe voice.ogg -b api
    """
    input_path = Path(input_path)

    # Create transcriber
    with console.status("[bold green]Loading model..."):
        transcriber = create_transcriber(
            backend=Backend(backend),
            model_size=ModelSize(model),
        )

    if batch or input_path.is_dir():
        _transcribe_batch(transcriber, input_path, output, output_format, language)
    else:
        _transcribe_single(transcriber, input_path, output, output_format, language)


def _transcribe_single(
        transcriber,
        input_path: Path,
        output: str | None,
        output_format: str,
        language: str | None
):
    """Transcribe single file."""
    with console.status(f"[bold green]Transcribing {input_path.name}..."):
        result = transcriber.transcribe(input_path, language)

    formatted = _format_result(result, output_format)

    if output:
        Path(output).write_text(formatted, encoding="utf-8")
        console.print(f"[green]✓[/green] Saved to {output}")
    else:
        console.print(formatted)

    # Stats
    console.print(f"\n[dim]Language: {result.language}, Duration: {result.duration:.1f}s[/dim]")


def _transcribe_batch(
        transcriber,
        input_dir: Path,
        output: str | None,
        output_format: str,
        language: str | None
):
    """Transcribe directory of files."""
    # Find all voice files
    files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in VOICE_EXTENSIONS
    ]

    if not files:
        console.print(f"[yellow]No voice files found in {input_dir}[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] voice files\n")

    results: dict[str, TranscriptionResult] = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
    ) as progress:
        task = progress.add_task("Transcribing...", total=len(files))

        for path, result in transcriber.transcribe_batch(files, language):
            results[path.name] = result
            progress.update(task, advance=1, description=f"[cyan]{path.name}[/cyan]")

    # Output
    if output_format == "json":
        output_data = {
            name: {
                "text": r.text,
                "language": r.language,
                "duration": r.duration,
            }
            for name, r in results.items()
        }
        formatted = json.dumps(output_data, ensure_ascii=False, indent=2)
    else:
        lines = []
        for name, r in results.items():
            lines.append(f"=== {name} ===")
            lines.append(r.text)
            lines.append("")
        formatted = "\n".join(lines)

    if output:
        Path(output).write_text(formatted, encoding="utf-8")
        console.print(f"\n[green]✓[/green] Saved to {output}")
    else:
        console.print(formatted)

    # Summary table
    _print_summary(results)


def _format_result(result: TranscriptionResult, fmt: str) -> str:
    if fmt == "json":
        return json.dumps({
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
        }, ensure_ascii=False, indent=2)
    elif fmt == "srt":
        return result.to_srt()
    else:
        return result.text


def _print_summary(results: dict[str, TranscriptionResult]):
    """Print summary table."""
    table = Table(title="\nSummary")
    table.add_column("File", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Language")
    table.add_column("Preview", max_width=40)

    total_duration = 0
    for name, r in results.items():
        preview = r.text[:40] + "..." if len(r.text) > 40 else r.text
        table.add_row(
            name,
            f"{r.duration:.1f}s",
            r.language,
            preview
        )
        total_duration += r.duration

    console.print(table)
    console.print(f"\n[bold]Total duration:[/bold] {total_duration:.1f}s ({total_duration / 60:.1f} min)")


@main.command()
def models():
    """List available models and their requirements."""
    table = Table(title="Available Models")
    table.add_column("Model")
    table.add_column("Parameters")
    table.add_column("VRAM")
    table.add_column("Speed*")
    table.add_column("Quality")

    table.add_row("tiny", "39M", "~1 GB", "~32x", "★☆☆☆☆")
    table.add_row("base", "74M", "~1 GB", "~16x", "★★☆☆☆")
    table.add_row("small", "244M", "~2 GB", "~6x", "★★★☆☆")
    table.add_row("medium", "769M", "~5 GB", "~2x", "★★★★☆")
    table.add_row("large-v3", "1550M", "~10 GB", "1x", "★★★★★")

    console.print(table)
    console.print("\n[dim]* Speed relative to large model. Use 'faster' backend for 4x speedup on CPU.[/dim]")


if __name__ == "__main__":
    main()