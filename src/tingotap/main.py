import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import asyncio
import json

from tingotap.config_gate import load_app_config
from tingotap.tap_assist import get_ai_response

app = typer.Typer(
    name="tingotap",
    help="A command-line assistant powered by various AI models via LiteLLM.",
    add_completion=False
)
console = Console()
current_app_config = None # Global app config for the session

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    global current_app_config
    if current_app_config is None:
        current_app_config = load_app_config() # UPDATED
        if not current_app_config:
            console.print("[bold red]Failed to load Tingotap configuration. Assistant may not work correctly.[/bold red]")

    if ctx.invoked_subcommand is None:
        console.print(Panel("[bold cyan]Welcome to Tingotap - Your AI Command Assistant![/bold cyan]\n"
                            "Type `tingotap ask \"Your question\"` to query the AI.", title="Tingotap"))

@app.command()
def ask(
    user_input: str = typer.Argument(..., help="The question or input for the AI."),
    profile: str = typer.Option(None, "--profile", "-p", help="Specify a model profile ID from the config (e.g., openai_gpt4o_mini)."),
    template: str = typer.Option("general_query", "--template", "-t", help="Specify a prompt template name from the config.")
):
    """
    Asks the AI a question using the specified or default model profile and prompt template.
    """
    global current_app_config
    if not current_app_config:
        console.print("[bold red]Tingotap configuration not loaded. Cannot process request.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"\n[dim]Your input:[/dim] {user_input}")
    if profile:
        console.print(f"[dim]Using profile override:[/dim] {profile}")
    if template != "general_query":
        console.print(f"[dim]Using prompt template:[/dim] {template}")

    async def _ask_async():
        ai_text_response = await get_ai_response(
            app_config=current_app_config,
            user_input_for_prompt=user_input,
            prompt_template_name=template,
            model_profile_override=profile
        )

        if ai_text_response:
            console.print(Panel(Markdown(ai_text_response), title="AI Response", border_style="green", expand=False))
        else:
            console.print(Panel("[yellow]The AI did not provide a response or an error occurred.[/yellow]", title="No Response", border_style="yellow"))

    asyncio.run(_ask_async())

@app.command()
def show_config(
    profile_name: str = typer.Option(None, "--profile", "-p", help="Display settings for a specific model profile.")
):
    """
    Displays the loaded Tingotap configuration or settings for a specific model profile.
    """
    global current_app_config
    from tingotap.config_gate import get_model_profile_settings # Local import for clarity

    if not current_app_config:
        console.print("[bold red]Tingotap configuration not loaded.[/bold red]")
        raise typer.Exit(code=1)

    if profile_name:
        settings = get_model_profile_settings(current_app_config, profile_name)
        if settings:
            console.print(Panel(json.dumps(settings, indent=2), title=f"Settings for Profile: '{profile_name}'", border_style="blue"))
        else:
            console.print(f"[yellow]Model profile '{profile_name}' not found in configuration.[/yellow]")
    else:
        console.print(Panel(json.dumps(current_app_config, indent=2), title="Loaded Tingotap Configuration", border_style="blue"))

if __name__ == "__main__":
    app()