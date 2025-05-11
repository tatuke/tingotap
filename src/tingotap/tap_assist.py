import litellm
from rich.console import Console
from rich.panel import Panel
# from rich.spinner import Spinner # LiteLLM has its own logging/status
import asyncio
from typing import Any, Dict, Optional, List

from tingotap.config_gate import get_model_profile_settings, get_prompt_template, load_app_config

console = Console()

# Optional: Configure LiteLLM global settings if needed
# litellm.set_verbose = True # Useful for debugging

async def get_ai_response(
    app_config: Dict[str, Any],
    user_input_for_prompt: str, # This is the value for the placeholder in the prompt template
    prompt_template_name: str = "general_query",
    model_profile_override: Optional[str] = None # To allow overriding the model profile for a specific call
) -> Optional[str]:
    """
    Gets a response from an AI model using LiteLLM based on the app_config.

    Args:
        app_config: The loaded application configuration.
        user_input_for_prompt: The actual text from the user (e.g., "What is Python?").
        prompt_template_name: The name of the prompt template from the config.
        model_profile_override: Optional. Name of the model profile (from config) to use.

    Returns:
        The AI's response as a string, or None if an error occurs.
    """
    if not app_config:
        console.print("[bold red]Application config not loaded. Cannot get AI response.[/bold red]")
        return None

    # 1. Get the prompt template
    prompt_template = get_prompt_template(app_config, prompt_template_name)
    if not prompt_template:
        console.print(f"[bold red]Prompt template '{prompt_template_name}' not found in config.[/bold red]")
        # Fallback: use user_input_for_prompt directly as the content of a single user message
        final_prompt_content = user_input_for_prompt
        # Determine model profile if template is missing
        profile_to_use_name = model_profile_override or app_config.get("metadata", {}).get("default_model_profile")
    else:
        # For simplicity, assuming prompt template "input" field is a simple string
        # with one placeholder like "{{query}}" or "{{topic}}"
        # A more robust templating engine (like Jinja2) could be used here.
        prompt_input_field = prompt_template.get("input", "User query: {{query}}") # Default if not specified
        # Infer the placeholder key (e.g., "query", "topic")
        # This is a simplistic way to find the placeholder
        placeholder_key = "query" # Default placeholder
        if "{{" in prompt_input_field and "}}" in prompt_input_field:
            start = prompt_input_field.find("{{") + 2
            end = prompt_input_field.find("}}", start)
            if start < end :
                placeholder_key = prompt_input_field[start:end].strip()

        final_prompt_content = prompt_input_field.replace(f"{{{{{placeholder_key}}}}}", user_input_for_prompt)
        profile_to_use_name = model_profile_override or prompt_template.get("metadata", {}).get("model_profile") \
                            or app_config.get("metadata", {}).get("default_model_profile")


    if not profile_to_use_name:
        console.print("[bold red]Could not determine a model profile to use.[/bold red]")
        return None

    # 2. Get model profile settings
    model_profile = get_model_profile_settings(app_config, profile_to_use_name)
    if not model_profile:
        # Error already printed by get_model_profile_settings
        return None

    litellm_model_name = model_profile.get("litellm_model_name")
    if not litellm_model_name:
        console.print(f"[bold red]Profile '{profile_to_use_name}' is missing 'litellm_model_name'.[/bold red]")
        return None

    # 3. Prepare parameters for LiteLLM
    messages = [{"role": "user", "content": final_prompt_content}] # Basic message structure

    # Extract LiteLLM specific params and general settings
    litellm_params = {
        "model": litellm_model_name,
        "messages": messages,
    }

    # Add api_base if present in profile (for Ollama or custom endpoints)
    if "litellm_api_base" in model_profile:
        litellm_params["api_base"] = model_profile["litellm_api_base"]

    # Add general settings (temperature, max_tokens, etc.)
    # LiteLLM will pass through supported parameters to the underlying model API
    general_settings = model_profile.get("settings", {})
    for key, value in general_settings.items():
        litellm_params[key] = value

    console.print(f"[cyan]Querying LiteLLM with model: {litellm_model_name} (Profile: {profile_to_use_name})[/cyan]")
    # For debugging the parameters sent to LiteLLM:
    # console.print(f"[grey50]LiteLLM params: {litellm_params}[/grey50]")

    try:
        # Use acompletion for async
        response = await litellm.acompletion(**litellm_params)

        # LiteLLM response structure is similar to OpenAI's
        # The main content is usually in response.choices[0].message.content
        ai_content = response.choices[0].message.content

        # You can also access usage information if needed
        # usage = response.usage # e.g., {'prompt_tokens': 20, 'completion_tokens': 150, 'total_tokens': 170}
        # console.print(f"[dim]Usage: {usage}[/dim]")

        return ai_content

    except litellm.RateLimitError as e:
        console.print(Panel(f"[bold red]LiteLLM Rate Limit Error:[/] {e}", title="API Error", border_style="red"))
    except litellm.APIConnectionError as e:
        console.print(Panel(f"[bold red]LiteLLM API Connection Error:[/] {e}", title="API Error", border_style="red"))
    except litellm.AuthenticationError as e:
        console.print(Panel(f"[bold red]LiteLLM Authentication Error:[/] {e}\n"
                            f"Ensure API key for '{litellm_model_name}' is set correctly as an environment variable.",
                            title="API Error", border_style="red"))
    except Exception as e:
        # Catch other LiteLLM errors or general errors
        console.print(Panel(f"[bold red]An unexpected error occurred with LiteLLM:[/] {e}", title="LiteLLM Error", border_style="red"))
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    return None


async def test_ai_connector():
    test_config = load_app_config() # Renamed function from config_manager
    if not test_config:
        console.print("[bold red]Failed to load test config for AI connector.[/bold red]")
        return

    console.print("\n--- Testing with 'ollama_default' profile (general_query) ---")
    # Ensure OPENAI_API_KEY (or other relevant keys) are set in your environment if testing cloud models
    # For Ollama, ensure Ollama server is running.
    response1 = await get_ai_response(test_config, "What is the capital of France?", prompt_template_name="general_query")
    if response1:
        console.print(Panel(response1, title="AI Response (Ollama Default)", border_style="green"))

    console.print("\n--- Testing with 'openai_gpt4o_mini' profile override (general_query) ---")
    # Make sure OPENAI_API_KEY is set in your environment!
    response2 = await get_ai_response(test_config, "Suggest three names for a new tech startup.", 
                                      prompt_template_name="general_query", model_profile_override="openai_gpt4o_mini")
    if response2:
        console.print(Panel(response2, title="AI Response (OpenAI GPT-4o-mini)", border_style="blue"))

    console.print("\n--- Testing with 'anthropic_claude3_haiku' profile (creative_writing template) ---")
    # Make sure ANTHROPIC_API_KEY is set in your environment!
    response3 = await get_ai_response(test_config, "a cat who dreams of flying", 
                                      prompt_template_name="creative_writing", 
                                      model_profile_override="anthropic_claude3_haiku") # Override if prompt doesn't specify it
    if response3:
        console.print(Panel(response3, title="AI Response (Claude 3 Haiku)", border_style="magenta"))


if __name__ == "__main__":
    # Make sure your environment variables for API keys are set before running this test!
    # e.g., export OPENAI_API_KEY="sk-..."
    # e.g., export ANTHROPIC_API_KEY="sk-ant-..."
    asyncio.run(test_ai_connector())