# Copyright Amity
"""
CLI-based Simulation Tool for tau-bench

This tool allows a human to act as the agent while an LLM simulates the user.
The human can respond with text or call tools with arguments.

This CLI is built on top of the SimulatorCore class, which provides the
decoupled simulation logic that can also be used via the web API.

Usage:
    python -m sigma.cli_simulator --env retail --user-model gpt-4o --user-provider openai

    Or with a custom persona:
    python -m sigma.cli_simulator --env retail --user-model gpt-4o --user-provider openai \
        --persona "You are John Smith. You want to cancel order #W1234567 because you ordered by mistake."
    
    Or with a persona file (created by persona_creator):
    python -m sigma.cli_simulator --env retail --user-model gpt-4o --user-provider openai \
        --persona-file sigma/personas/my_persona_retail.json
"""

import argparse
import json
import os
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown

from sigma.types import Action, RESPOND_ACTION_NAME

from sigma.env_registry import (
    get_environment_config,
    list_environments,
    get_all_environment_configs,
    EnvironmentConfig,
)
from sigma.simulator_core import (
    SimulatorCore,
    ActionResult,
    ConversationEntry,
    load_persona_file,
)


class InteractionMode(Enum):
    """Mode of interaction with the simulator."""
    MENU = "menu"
    NATURAL_LANGUAGE = "natural_language"


console = Console()

# Sentinel value for going back to previous menu
BACK_SENTINEL = "__BACK__"


class CLISimulator:
    """
    CLI-based simulator where human acts as agent, LLM acts as user.
    
    This class provides a rich terminal UI on top of the SimulatorCore.
    """

    def __init__(
        self,
        env_name: str,
        user_model: str,
        user_provider: str,
        agent_model: Optional[str] = None,
        agent_provider: Optional[str] = None,
        persona: Optional[str] = None,
        persona_file: Optional[str] = None,
        task_index: Optional[int] = None,
        task_split: str = "test",
        generate_scenario: bool = False,
    ):
        self.env_name = env_name
        self.user_model = user_model
        self.user_provider = user_provider
        self.agent_model = agent_model or user_model
        self.agent_provider = agent_provider or user_provider
        self.persona = persona
        self.persona_file = persona_file
        self.task_index = task_index
        self.task_split = task_split
        self.generate_scenario = generate_scenario
        
        # Load persona data if file provided
        persona_data = None
        if persona_file:
            persona_data = load_persona_file(persona_file)
        
        # Create the core simulator
        self.core = SimulatorCore(
            env_name=env_name,
            user_model=user_model,
            user_provider=user_provider,
            agent_model=agent_model,
            agent_provider=agent_provider,
            persona=persona,
            persona_data=persona_data,
            task_index=task_index,
            task_split=task_split,
            generate_scenario=generate_scenario,
        )
        
        # Convenience references
        self.tools_info = self.core.tools_raw
        self.interaction_mode: InteractionMode = InteractionMode.MENU
    
    @property
    def env(self):
        """Access to underlying environment (for backward compatibility)."""
        return self.core.env
    
    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as list of dicts."""
        return [
            {
                "role": e.role,
                "content": e.content,
                "tool_call": e.tool_call,
            }
            for e in self.core.conversation_history
        ]

    def display_welcome(self):
        """Display welcome message and setup info."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]🎮 tau-bench CLI Simulator[/bold blue]\n\n"
            "You are the [bold green]AGENT[/bold green]. An LLM will simulate the [bold yellow]USER[/bold yellow].\n"
            "Respond to the user or call tools to complete their requests.\n\n"
            "[dim]Tip: Type 'b' to go back in any submenu[/dim]",
            title="Welcome",
            border_style="blue"
        ))
        
        # Display environment info
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Key", style="cyan")
        info_table.add_column("Value", style="white")
        info_table.add_row("Environment", self.env_name)
        info_table.add_row("User Model", f"{self.user_model} ({self.user_provider})")
        info_table.add_row("Agent Assist Model", f"{self.agent_model} ({self.agent_provider})")
        info_table.add_row("Available Tools", str(len(self.tools_info)))
        
        # Show if scenario was auto-generated
        if self.generate_scenario and self.core.generated_scenario:
            info_table.add_row("Scenario", "[bold magenta]AUTO-GENERATED[/bold magenta]")
        
        console.print(info_table)
        console.print()

    def display_persona(self, persona: str):
        """Display the user persona (for reference)."""
        # Determine if this is a generated scenario
        is_generated = self.generate_scenario and self.core.generated_scenario
        
        title = "[bold yellow]User Persona"
        if is_generated:
            title += " (AUTO-GENERATED)"
        title += "[/bold yellow]"
        
        console.print(Panel(
            persona,
            title=title,
            border_style="yellow",
            subtitle="This is what the simulated user is trying to accomplish"
        ))
        console.print()

    def display_wiki(self):
        """Display the environment wiki/policy."""
        console.print(Panel(
            Markdown(self.env.wiki),
            title="[bold cyan]Agent Policy / Wiki[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

    def display_tools_list(self):
        """Display available tools in a table."""
        table = Table(title="Available Tools", show_lines=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Tool Name", style="green")
        table.add_column("Description", style="white", max_width=60)
        
        for i, tool in enumerate(self.tools_info):
            func_info = tool.get("function", {})
            name = func_info.get("name", "Unknown")
            desc = func_info.get("description", "No description")
            # Truncate description if too long
            if len(desc) > 100:
                desc = desc[:97] + "..."
            table.add_row(str(i), name, desc)
        
        console.print(table)
        console.print()

    def display_tool_details(self, tool_index: int):
        """Display detailed info about a specific tool."""
        if tool_index < 0 or tool_index >= len(self.tools_info):
            console.print("[red]Invalid tool index[/red]")
            return
            
        tool = self.tools_info[tool_index]
        func_info = tool.get("function", {})
        
        console.print(Panel(
            f"[bold]{func_info.get('name', 'Unknown')}[/bold]\n\n"
            f"{func_info.get('description', 'No description')}\n\n"
            f"[cyan]Parameters:[/cyan]\n"
            f"{json.dumps(func_info.get('parameters', {}), indent=2)}",
            title="Tool Details",
            border_style="green"
        ))

    def display_user_message(self, message: str):
        """Display a message from the user."""
        console.print(Panel(
            message,
            title="[bold yellow]👤 User[/bold yellow]",
            border_style="yellow"
        ))
        # Note: conversation history is managed by the core

    def display_agent_message(self, message: str):
        """Display a message from the agent."""
        console.print(Panel(
            message,
            title="[bold green]🤖 Agent (You)[/bold green]",
            border_style="green"
        ))
        # Note: conversation history is managed by the core

    def display_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Display a tool call."""
        console.print(Panel(
            f"[cyan]Tool:[/cyan] {tool_name}\n"
            f"[cyan]Arguments:[/cyan]\n{json.dumps(args, indent=2)}",
            title="[bold magenta]🔧 Tool Call[/bold magenta]",
            border_style="magenta"
        ))
        # Note: conversation history is managed by the core

    def display_tool_result(self, result: str):
        """Display tool execution result."""
        # Try to format as JSON if possible
        try:
            parsed = json.loads(result)
            formatted = json.dumps(parsed, indent=2)
            content = Syntax(formatted, "json", theme="monokai", line_numbers=False)
        except json.JSONDecodeError:
            content = result
            
        console.print(Panel(
            content,
            title="[bold blue]📋 Tool Result[/bold blue]",
            border_style="blue"
        ))
        # Note: conversation history is managed by the core

    def _build_agent_context(self) -> str:
        """Build context string for agent LLM assistance."""
        return self.core._build_agent_context()

    def _generate_agent_response(self, prompt: str) -> str:
        """Generate an agent response using LLM based on user's prompt."""
        with console.status("[bold green]Generating response...[/bold green]"):
            return self.core.generate_response(prompt)

    def _get_response_to_user(self) -> Union[str, None]:
        """Get response to user - either direct text or LLM-generated."""
        console.print("\n[bold]How would you like to respond?[/bold]")
        console.print("  [cyan]1[/cyan] - Type exact response")
        console.print("  [cyan]2[/cyan] - Describe what to say (LLM generates)")
        console.print("  [cyan]b[/cyan] - Back to main menu")
        console.print()
        
        while True:
            choice = Prompt.ask("Your choice", default="2")
            
            if choice.lower() == "b":
                return BACK_SENTINEL
            
            elif choice == "1":
                response = Prompt.ask("[green]Enter your exact response[/green]")
                if response.lower() == "b":
                    return BACK_SENTINEL
                return response
                
            elif choice == "2":
                prompt = Prompt.ask("[green]Describe what you want to say[/green]")
                if prompt.lower() == "b":
                    return BACK_SENTINEL
                
                generated = self._generate_agent_response(prompt)
                if generated is None:
                    console.print("[yellow]Failed to generate. Try again or type exact response.[/yellow]")
                    continue
                
                # Show generated response and ask for confirmation
                console.print(Panel(
                    generated,
                    title="[bold cyan]Generated Response (Preview)[/bold cyan]",
                    border_style="cyan"
                ))
                
                confirm = Prompt.ask(
                    "Use this response?",
                    choices=["y", "n", "e", "b"],
                    default="y"
                )
                
                if confirm == "y":
                    return generated
                elif confirm == "e":
                    # Edit the response
                    edited = Prompt.ask("[green]Edit response[/green]", default=generated)
                    return edited
                elif confirm == "b":
                    return BACK_SENTINEL
                # If 'n', loop continues to try again
                
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")

    def get_agent_action(self) -> Action:
        """Get the agent's action from human input."""
        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  [cyan]1[/cyan] - Respond to user")
        console.print("  [cyan]2[/cyan] - Call a tool")
        console.print("  [cyan]3[/cyan] - View available tools")
        console.print("  [cyan]4[/cyan] - View tool details")
        console.print("  [cyan]5[/cyan] - View wiki/policy")
        console.print("  [cyan]6[/cyan] - View conversation history")
        console.print("  [cyan]n[/cyan] - Switch to natural language mode")
        console.print("  [cyan]q[/cyan] - Quit simulation")
        console.print()
        
        while True:
            choice = Prompt.ask("Your choice", default="1")
            
            if choice == "q":
                console.print("[yellow]Ending simulation...[/yellow]")
                sys.exit(0)
            
            elif choice.lower() == "n":
                self.interaction_mode = InteractionMode.NATURAL_LANGUAGE
                console.print("[green]Switched to natural language mode.[/green]")
                console.print("[dim]Tip: Type 'menu' to switch back.[/dim]")
                return self.get_agent_action_natural_language()
                
            elif choice == "1":
                # Text response with LLM assistance
                response = self._get_response_to_user()
                if response == BACK_SENTINEL:
                    continue  # Show menu again
                return Action(name=RESPOND_ACTION_NAME, kwargs={"content": response})
                
            elif choice == "2":
                # Tool call
                result = self._get_tool_call()
                if result == BACK_SENTINEL:
                    continue  # Show menu again
                return result
                
            elif choice == "3":
                self.display_tools_list()
                
            elif choice == "4":
                tool_input = Prompt.ask("Enter tool number to view details (or 'b' to go back)")
                if tool_input.lower() == "b":
                    continue
                try:
                    tool_idx = int(tool_input)
                    self.display_tool_details(tool_idx)
                except ValueError:
                    console.print("[red]Invalid number[/red]")
                
            elif choice == "5":
                self.display_wiki()
                
            elif choice == "6":
                self._display_conversation_history()
                
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")

    def _get_tool_call(self) -> Union[Action, str]:
        """Get tool call details from human input. Returns BACK_SENTINEL to go back."""
        self.display_tools_list()
        
        tool_input = Prompt.ask("Enter tool number to call (or 'b' to go back)")
        if tool_input.lower() == "b":
            return BACK_SENTINEL
        
        try:
            tool_idx = int(tool_input)
        except ValueError:
            console.print("[red]Invalid number. Returning to menu.[/red]")
            return BACK_SENTINEL
            
        if tool_idx < 0 or tool_idx >= len(self.tools_info):
            console.print("[red]Invalid tool index. Returning to menu.[/red]")
            return BACK_SENTINEL
        
        tool = self.tools_info[tool_idx]
        func_info = tool.get("function", {})
        tool_name = func_info.get("name")
        params = func_info.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        
        console.print(f"\n[cyan]Calling tool:[/cyan] [bold]{tool_name}[/bold]")
        console.print(f"[dim]{func_info.get('description', '')}[/dim]")
        console.print("[dim]Enter 'b' for any parameter to cancel and go back[/dim]\n")
        
        # Collect arguments
        args = {}
        for param_name, param_info in properties.items():
            is_required = param_name in required
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            enum_values = param_info.get("enum")
            
            prompt_text = f"[{'bold' if is_required else 'dim'}]{param_name}[/]"
            if enum_values:
                prompt_text += f" ({'/'.join(enum_values)})"
            prompt_text += f" [{param_type}]"
            if param_desc:
                console.print(f"  [dim]{param_desc}[/dim]")
            
            if is_required:
                value = Prompt.ask(f"  {prompt_text}")
            else:
                value = Prompt.ask(f"  {prompt_text} (optional)", default="")
                if not value:
                    continue
            
            # Check for back command
            if value.lower() == "b":
                return BACK_SENTINEL
            
            # Type conversion
            if param_type == "integer":
                try:
                    value = int(value)
                except ValueError:
                    console.print(f"[yellow]Warning: Could not convert to integer, using string[/yellow]")
            elif param_type == "number":
                try:
                    value = float(value)
                except ValueError:
                    console.print(f"[yellow]Warning: Could not convert to number, using string[/yellow]")
            elif param_type == "boolean":
                value = value.lower() in ("true", "yes", "1")
            elif param_type == "array":
                # Try to parse as JSON array
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    # Split by comma
                    value = [v.strip() for v in value.split(",")]
            elif param_type == "object":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    console.print(f"[yellow]Warning: Could not parse as JSON object[/yellow]")
            
            args[param_name] = value
        
        # Confirm
        console.print(f"\n[cyan]Arguments:[/cyan]")
        console.print(json.dumps(args, indent=2))
        confirm = Prompt.ask("Confirm tool call?", choices=["y", "n", "b"], default="y")
        
        if confirm == "n" or confirm == "b":
            return BACK_SENTINEL
        
        return Action(name=tool_name, kwargs=args)

    def _display_conversation_history(self):
        """Display the conversation history."""
        if not self.conversation_history:
            console.print("[dim]No conversation history yet.[/dim]")
            return
            
        console.print(Panel.fit("[bold]Conversation History[/bold]", border_style="white"))
        for entry in self.conversation_history:
            role = entry.get("role", "unknown")
            if role == "user":
                console.print(f"[yellow]👤 User:[/yellow] {entry.get('content', '')}")
            elif role == "agent":
                if "tool_call" in entry:
                    tc = entry["tool_call"]
                    console.print(f"[green]🤖 Agent:[/green] [magenta]Tool: {tc['name']}({tc['arguments']})[/magenta]")
                else:
                    console.print(f"[green]🤖 Agent:[/green] {entry.get('content', '')}")
            elif role == "tool":
                result = entry.get("content", "")
                if len(result) > 100:
                    result = result[:97] + "..."
                console.print(f"[blue]📋 Tool Result:[/blue] {result}")
        console.print()

    def _get_tools_summary(self) -> str:
        """Get a concise summary of available tools for LLM context."""
        return self.core._get_tools_summary()

    def _parse_natural_language_action(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Use LLM to parse natural language into an action."""
        with console.status("[bold green]Interpreting your request...[/bold green]"):
            parsed = self.core.parse_natural_language_action(user_input)
        
        if parsed is None:
            console.print("[red]Error parsing your request[/red]")
            return None
        
        return parsed

    def _display_action_confirmation(self, parsed_action: Dict[str, Any]) -> bool:
        """Display the parsed action and ask for confirmation."""
        action_type = parsed_action.get("action_type", "unknown")
        
        if action_type == "respond":
            console.print(Panel(
                parsed_action.get("content", ""),
                title="[bold cyan]📝 Proposed Response to User[/bold cyan]",
                border_style="cyan"
            ))
        elif action_type == "tool_call":
            tool_name = parsed_action.get("tool_name", "unknown")
            arguments = parsed_action.get("arguments", {})
            console.print(Panel(
                f"[cyan]Tool:[/cyan] {tool_name}\n"
                f"[cyan]Arguments:[/cyan]\n{json.dumps(arguments, indent=2)}",
                title="[bold magenta]🔧 Proposed Tool Call[/bold magenta]",
                border_style="magenta"
            ))
        elif action_type == "view_wiki":
            console.print("[cyan]Action: View wiki/policy[/cyan]")
            return True  # No confirmation needed for view actions
        elif action_type == "view_tools":
            console.print("[cyan]Action: View tools list[/cyan]")
            return True  # No confirmation needed for view actions
        elif action_type == "quit":
            console.print("[yellow]Action: Quit simulation[/yellow]")
        else:
            console.print(f"[red]Unknown action type: {action_type}[/red]")
            return False
        
        return Confirm.ask("Proceed with this action?", default=True)

    def get_agent_action_natural_language(self) -> Optional[Action]:
        """Get agent action using natural language input."""
        # Display compact tools reference
        console.print("\n[dim]Available tools:[/dim]")
        for i, tool in enumerate(self.tools_info):
            func_info = tool.get("function", {})
            name = func_info.get("name", "Unknown")
            console.print(f"  [dim]{i}. {name}[/dim]")
        
        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("[dim]Describe your action in natural language, or type:[/dim]")
        console.print("[dim]  'tools' - view all tools, 'wiki' - view policy, 'menu' - switch to menu mode, 'q' - quit[/dim]")
        console.print()
        
        while True:
            user_input = Prompt.ask("[green]Your action[/green]")
            
            # Handle special commands
            if user_input.lower() == "q":
                console.print("[yellow]Ending simulation...[/yellow]")
                sys.exit(0)
            elif user_input.lower() == "tools":
                self.display_tools_list()
                continue
            elif user_input.lower() == "wiki":
                self.display_wiki()
                continue
            elif user_input.lower() == "menu":
                self.interaction_mode = InteractionMode.MENU
                console.print("[green]Switched to menu mode.[/green]")
                return self.get_agent_action()  # Fall back to menu mode
            elif user_input.lower() == "history":
                self._display_conversation_history()
                continue
            
            # Parse natural language
            parsed = self._parse_natural_language_action(user_input)
            
            if parsed is None:
                console.print("[yellow]Could not understand. Please try again or be more specific.[/yellow]")
                continue
            
            # Confirm the action
            if not self._display_action_confirmation(parsed):
                console.print("[yellow]Action cancelled. Try again.[/yellow]")
                continue
            
            # Execute the parsed action
            action_type = parsed.get("action_type", "unknown")
            
            if action_type == "respond":
                return Action(name=RESPOND_ACTION_NAME, kwargs={"content": parsed.get("content", "")})
            
            elif action_type == "tool_call":
                tool_name = parsed.get("tool_name")
                arguments = parsed.get("arguments", {})
                return Action(name=tool_name, kwargs=arguments)
            
            elif action_type == "view_wiki":
                self.display_wiki()
                continue
            
            elif action_type == "view_tools":
                self.display_tools_list()
                continue
            
            elif action_type == "quit":
                console.print("[yellow]Ending simulation...[/yellow]")
                sys.exit(0)
            
            else:
                console.print(f"[red]Unknown action type: {action_type}[/red]")
                continue

    def _select_interaction_mode(self):
        """Let user select interaction mode."""
        console.print("\n[bold]Select interaction mode:[/bold]")
        console.print("  [cyan]1[/cyan] - Menu-based navigation (guided step-by-step)")
        console.print("  [cyan]2[/cyan] - Natural language (describe actions freely)")
        console.print()
        
        while True:
            choice = Prompt.ask("Your choice", choices=["1", "2"], default="1")
            if choice == "1":
                self.interaction_mode = InteractionMode.MENU
                console.print("[green]Using menu-based navigation.[/green]")
                break
            elif choice == "2":
                self.interaction_mode = InteractionMode.NATURAL_LANGUAGE
                console.print("[green]Using natural language mode.[/green]")
                console.print("[dim]Tip: Describe what you want to do, and I'll interpret it.[/dim]")
                console.print("[dim]You can switch to menu mode anytime by typing 'menu'.[/dim]")
                break

    def run(self):
        """Run the CLI simulation."""
        self.display_welcome()
        
        # Select interaction mode
        self._select_interaction_mode()
        
        # Determine persona to display
        persona = self.core.current_persona
        
        self.display_persona(persona)
        
        # Ask if user wants to see wiki
        show_wiki = Prompt.ask("Show agent policy/wiki?", choices=["y", "n"], default="n")
        if show_wiki == "y":
            self.display_wiki()
        
        console.print(Panel.fit(
            "[bold]Simulation Starting![/bold]\n"
            "The user will send the first message.",
            border_style="green"
        ))
        console.print()
        
        # Start simulation using the core
        initial_message = self.core.start()
        
        # Display initial user message
        self.display_user_message(initial_message)
        
        # Main interaction loop
        done = False
        while not done:
            # Get agent action based on interaction mode
            if self.interaction_mode == InteractionMode.NATURAL_LANGUAGE:
                action = self.get_agent_action_natural_language()
            else:
                action = self.get_agent_action()
            
            if action is None:
                continue  # Action was cancelled or invalid
            
            if action.name == RESPOND_ACTION_NAME:
                # Text response to user
                self.display_agent_message(action.kwargs["content"])
                result = self.core.respond_to_user(action.kwargs["content"])
                
                if result.done:
                    done = True
                    self._display_final_result(result)
                else:
                    self.display_user_message(result.observation)
            else:
                # Tool call
                self.display_tool_call(action.name, action.kwargs)
                result = self.core.call_tool(action.name, action.kwargs)
                self.display_tool_result(result.observation)
                
                if result.done:
                    done = True
                    self._display_final_result(result)

    def _display_final_result(self, result: ActionResult):
        """Display the final result of the simulation."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold]Simulation Complete![/bold]",
            border_style="green"
        ))
        
        # Get the full result from core
        final_result = self.core.get_final_result()
        
        # Display reward info
        if result.reward is not None:
            result_table = Table(title="Results", show_header=False)
            result_table.add_column("Metric", style="cyan")
            result_table.add_column("Value", style="white")
            result_table.add_row("Reward", f"[{'green' if result.reward == 1.0 else 'red'}]{result.reward}[/]")
            
            if result.reward_info:
                if "r_actions" in result.reward_info:
                    result_table.add_row("Actions Correct", str(result.reward_info["r_actions"]))
                if "r_outputs" in result.reward_info:
                    result_table.add_row("Outputs Correct", str(result.reward_info["r_outputs"]))
                    if "outputs" in result.reward_info:
                        for output, found in result.reward_info["outputs"].items():
                            result_table.add_row(f"  Output: {output}", "✓" if found else "✗")
            
            console.print(result_table)
        
        # Display expected actions for learning
        if final_result.get("expected_actions"):
            console.print("\n[bold cyan]Expected Actions (Ground Truth):[/bold cyan]")
            for i, action in enumerate(final_result["expected_actions"]):
                console.print(f"  {i+1}. [green]{action['name']}[/green]({json.dumps(action['kwargs'])})")


def main():
    # Get available environments from registry
    available_envs = list_environments()
    env_configs = get_all_environment_configs()
    env_descriptions = ", ".join([f"{name} ({cfg.display_name})" for name, cfg in env_configs.items()])
    
    parser = argparse.ArgumentParser(
        description="CLI-based Simulation for tau-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Use a random task from retail environment
  python -m sigma --env retail --user-model gpt-4o --user-provider openai

  # Auto-generate a NEW scenario (inspired by existing tasks)
  python -m sigma --env retail --user-model gpt-4o --user-provider openai --generate-scenario

  # Use a specific task
  python -m sigma --env retail --user-model gpt-4o --user-provider openai --task-index 5

  # Use a custom persona (text)
  python -m sigma --env retail --user-model gpt-4o --user-provider openai \\
      --persona "You are John. You want to cancel order #W1234567."

  # Use a persona file (created by persona_creator)
  python -m sigma --env retail --user-model gpt-4o --user-provider openai \\
      --persona-file sigma/personas/my_persona_retail.json

  # Create a new persona first
  python -m sigma.persona_creator --env retail --model gpt-4o --provider openai

Available environments: {env_descriptions}
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=available_envs,
        default=available_envs[0] if available_envs else "retail",
        help=f"Environment to use (default: {available_envs[0] if available_envs else 'retail'})"
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="LLM model for user simulation (default: gpt-4o)"
    )
    parser.add_argument(
        "--user-provider",
        type=str,
        default="openai",
        help="LLM provider for user simulation (default: openai)"
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="LLM model for agent response generation (default: same as user-model)"
    )
    parser.add_argument(
        "--agent-provider",
        type=str,
        default=None,
        help="LLM provider for agent response generation (default: same as user-provider)"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Custom persona/instruction for the simulated user (text)"
    )
    parser.add_argument(
        "--persona-file",
        type=str,
        default=None,
        help="Path to a persona JSON file (created by persona_creator)"
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Specific task index to use (default: random)"
    )
    parser.add_argument(
        "--task-split",
        type=str,
        choices=["test", "train", "dev"],
        default="test",
        help="Task split to use (default: test)"
    )
    parser.add_argument(
        "--generate-scenario",
        action="store_true",
        help="Auto-generate a new scenario inspired by existing tasks (creates new user, orders, etc.)"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    
    args = parser.parse_args()
    
    # Check for rich library
    try:
        from rich.console import Console
    except ImportError:
        print("Error: 'rich' library is required. Install with: pip install rich")
        sys.exit(1)
    
    # List tasks mode
    if args.list_tasks:
        try:
            env_config = get_environment_config(args.env)
            if env_config.tasks_loader:
                tasks = env_config.tasks_loader(args.task_split)
            else:
                console.print(f"[red]No tasks loader for environment: {args.env}[/red]")
                sys.exit(1)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        
        console = Console()
        table = Table(title=f"Available Tasks ({args.env} - {args.task_split})")
        table.add_column("#", style="cyan", width=4)
        table.add_column("ID", style="green", width=6)
        table.add_column("Reason for Call", style="white", max_width=80)
        
        for i, task in enumerate(tasks):
            if isinstance(task, dict):
                # Check if it's new retail_t2 format (has user_scenario)
                if "user_scenario" in task:
                    task_id = str(task.get("id", "N/A"))
                    user_scenario = task.get("user_scenario", {})
                    instructions = user_scenario.get("instructions", {})
                    reason = instructions.get("reason_for_call", "N/A")
                    instruction = reason
                else:
                    # Legacy format
                    task_id = str(task.get("id", i))
                    instruction = task.get("instruction", "N/A")
            else:
                task_id = str(i)
                instruction = task.instruction
            
            if len(instruction) > 80:
                instruction = instruction[:77] + "..."
            table.add_row(str(i), task_id, instruction)
        
        console.print(table)
        sys.exit(0)
    
    # Validate persona-file exists
    if args.persona_file and not os.path.exists(args.persona_file):
        console.print(f"[red]Error: Persona file not found: {args.persona_file}[/red]")
        sys.exit(1)
    
    # Run simulator
    simulator = CLISimulator(
        env_name=args.env,
        user_model=args.user_model,
        user_provider=args.user_provider,
        agent_model=args.agent_model,
        agent_provider=args.agent_provider,
        persona=args.persona,
        persona_file=args.persona_file,
        task_index=args.task_index,
        task_split=args.task_split,
        generate_scenario=args.generate_scenario,
    )
    
    try:
        simulator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
