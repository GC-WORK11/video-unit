"""Generate test contracts from text prompts."""

import asyncio

import httpx
import typer
import yaml

from videounit_cli.utils.backend import get_backend_url
from videounit_cli.utils.output import console, print_error, print_success


app = typer.Typer(help="Generate a test contract from a text prompt")


@app.command()
def generate_contract(
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Text description of expected video behavior",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for contract YAML (default: contract.yaml)",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Backend URL",
    ),
    view: bool = typer.Option(
        False,
        "--view",
        "-v",
        help="View generated contract without saving",
    ),
) -> None:
    """Generate a test contract from a text prompt using VLM.

    Example:
        videounit generate-contract --prompt "A red ball rolls into a glass"
        videounit generate-contract -p "A car drives from left to right" -o my_contract.yaml
    """
    backend_url = backend or get_backend_url()

    console.print(f"[cyan]Generating contract from prompt:[/cyan]")
    console.print(f"  {prompt}")

    async def _generate():
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{backend_url}/api/videounit/contract/generate"

            try:
                response = await client.post(
                    url,
                    json={"prompt": prompt, "provider": "minimax"},
                )

                response.raise_for_status()
                data = response.json()

                return data

            except httpx.HTTPError as e:
                print_error(f"Failed to generate contract: {e}")
                raise typer.Exit(1)

    result = asyncio.run(_generate())

    contract_yaml = result.get("contract_yaml", "")

    if view or not output:
        console.print()
        console.print("[cyan]Generated Contract:[/cyan]")
        console.print()
        console.print(contract_yaml)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(contract_yaml)
        print_success(f"Contract saved: {output_path}")

    console.print()
    print_success(
        f"Generated {result.get('assertions', 0)} assertions for "
        f"{len(result.get('objects', []))} objects"
    )


if __name__ == "__main__":
    app()
