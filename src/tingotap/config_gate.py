import json
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
# We might not use AIConfigRuntime directly if LiteLLM simplifies things,
# but its structure can still be loaded with json.load
# from aiconfig import AIConfigRuntime

CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "configs" / "tap_config.json"

console = Console()

def load_app_config() -> Optional[Dict[str, Any]]:
    """Loads the application config from the specified JSON file."""
    try:
        if not CONFIG_FILE_PATH.exists():
            console.print(f"[bold red]Error: Configuration file not found at {CONFIG_FILE_PATH}[/bold red]")
            # You could create a default config here if it doesn't exist
            return None
        with open(CONFIG_FILE_PATH, 'r') as f:
            config_data = json.load(f)
        return config_data
    except Exception as e:
        console.print(Panel(f"[bold red]Error loading configuration:[/] {e}", title="Config Error", border_style="red"))
        return None

def get_model_profile_settings(app_config: Dict[str, Any], profile_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves the settings for a given model profile_name or the default model profile.
    """
    if not app_config or "metadata" not in app_config or "models" not in app_config["metadata"]:
        console.print("[bold red]Invalid application configuration.[/bold red]")
        return None

    models_config = app_config["metadata"]["models"]

    resolved_profile_name = profile_name
    if not resolved_profile_name:
        resolved_profile_name = app_config["metadata"].get("default_model_profile")

    if not resolved_profile_name:
        console.print("[bold red]Error: No model profile specified and no default_model_profile found in config.[/bold red]")
        return None

    if resolved_profile_name in models_config:
        return models_config[resolved_profile_name]
    else:
        console.print(f"[bold red]Error: Model profile '{resolved_profile_name}' not found in config.metadata.models.[/bold red]")
        return None

def get_prompt_template(app_config: Dict[str, Any], prompt_name: str) -> Optional[Dict[str, Any]]:
    if not app_config or "prompts" not in app_config:
        return None
    for p_template in app_config["prompts"]:
        if p_template.get("name") == prompt_name:
            return p_template
    return None


if __name__ == "__main__":
    # Example usage
    current_config = load_app_config()
    if current_config:
        console.print(f"Successfully loaded config: {current_config.get('name')}")

        default_profile_key = current_config.get("metadata", {}).get("default_model_profile", "N/A")
        console.print(f"Default model profile key: {default_profile_key}")

        settings = get_model_profile_settings(current_config) # Get default
        if settings:
            console.print(f"\nSettings for default profile ({default_profile_key}):")
            console.print(settings)

        openai_settings = get_model_profile_settings(current_config, "openai_gpt4o_mini")
        if openai_settings:
            console.print("\nSettings for 'openai_gpt4o_mini':")
            console.print(openai_settings)

        gen_query_prompt = get_prompt_template(current_config, "general_query")
        if gen_query_prompt:
            console.print("\n'general_query' prompt template:")
            console.print(gen_query_prompt)