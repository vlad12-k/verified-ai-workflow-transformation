"""Command-line interface for VAIT."""

import typer

app = typer.Typer(
    name="vait",
    help="Verified AI Workflow Transformation Engine.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Verified AI Workflow Transformation Engine."""
    pass


@app.command()
def version() -> None:
    """Display the installed VAIT version."""
    typer.echo("VAIT 0.1.0")


if __name__ == "__main__":
    app()
