# Copyright Amity
"""
Persona Creator for tau-bench Simulator

This tool creates custom personas with generated data that can be used
in the CLI simulator. It generates realistic user profiles, orders/reservations
that fit the scenario the user wants to simulate.

Usage:
    python -m sigma.persona_creator --env retail --model gpt-4o --provider openai
"""

import argparse
import importlib
import json
import os
import random
import string
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax

from litellm import completion

from sigma.env_registry import (
    get_environment_config,
    list_environments,
    get_all_environment_configs,
    EnvironmentConfig,
)

console = Console()

# Directory to store generated personas
PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "personas")


def ensure_personas_dir():
    """Ensure the personas directory exists."""
    os.makedirs(PERSONAS_DIR, exist_ok=True)


def generate_id(length: int = 8) -> str:
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_order_id() -> str:
    """Generate a retail order ID."""
    return f"#W{random.randint(1000000, 9999999)}"


def generate_reservation_id() -> str:
    """Generate an airline reservation ID."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=6))


# =============================================================================
# Data Schema Templates (for LLM context)
# =============================================================================

RETAIL_USER_SCHEMA = """
{
    "user_id": "firstname_lastname_1234",
    "name": { "first_name": "John", "last_name": "Doe" },
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "email": "john.doe1234@example.com",
    "payment_methods": {
        "credit_card_1234567": {
            "source": "credit_card",
            "brand": "visa",
            "last_four": "1234",
            "id": "credit_card_1234567"
        },
        "gift_card_7654321": {
            "source": "gift_card",
            "balance": 150.00,
            "id": "gift_card_7654321"
        },
        "paypal_9876543": {
            "source": "paypal",
            "id": "paypal_9876543"
        }
    },
    "orders": ["#W1234567", "#W7654321"]
}
"""

RETAIL_ORDER_SCHEMA = """
{
    "order_id": "#W1234567",
    "user_id": "john_doe_1234",
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "items": [
        {
            "name": "T-Shirt",
            "product_id": "9523456873",
            "item_id": "1234567890",
            "price": 49.99,
            "options": {
                "color": "blue",
                "size": "M",
                "material": "cotton",
                "style": "crew neck"
            }
        }
    ],
    "fulfillments": [
        {
            "tracking_id": ["123456789012"],
            "item_ids": ["1234567890"]
        }
    ],
    "status": "delivered",  // Can be: pending, processed, delivered, cancelled
    "payment_history": [
        {
            "transaction_type": "payment",
            "amount": 49.99,
            "payment_method_id": "credit_card_1234567"
        }
    ]
}
"""

RETAIL_PRODUCTS_INFO = """
Available product types:
- T-Shirt (options: color, size, material, style)
- Jeans (options: color, size, fit, material)  
- Sneakers (options: color, size, material)
- Backpack (options: color, size, material, compartments)
- Water Bottle (options: capacity, material, color)
- Office Chair (options: material, color, armrest, backrest height)
- Laptop (options: brand, screen size, processor, RAM, storage)
- Headphones (options: type, connectivity, color)
- Smart Watch (options: brand, color, band material, screen size)
- Electric Kettle (options: capacity, material, color)
- Mechanical Keyboard (options: switch type, backlight, size)
- Smart Thermostat (options: compatibility, color)
- Action Camera (options: resolution, waterproof, color)
- Bookshelf (options: material, color, height)
- Desk Lamp (options: color, brightness, power source)
"""

AIRLINE_USER_SCHEMA = """
{
    "user_id": "firstname_lastname_1234",
    "name": { "first_name": "John", "last_name": "Doe" },
    "address": {
        "address1": "123 Main Street",
        "address2": "Apt 4B",
        "city": "New York",
        "country": "USA",
        "state": "NY",
        "zip": "10001"
    },
    "email": "john.doe1234@example.com",
    "dob": "1985-06-15",
    "payment_methods": {
        "credit_card_1234567": {
            "source": "credit_card",
            "brand": "visa",
            "last_four": "1234",
            "id": "credit_card_1234567"
        },
        "gift_card_7654321": {
            "source": "gift_card",
            "amount": 500,
            "id": "gift_card_7654321"
        },
        "certificate_9876543": {
            "source": "certificate",
            "amount": 250,
            "id": "certificate_9876543"
        }
    },
    "saved_passengers": [
        { "first_name": "Jane", "last_name": "Doe", "dob": "1987-03-20" }
    ],
    "membership": "gold",  // Can be: regular, silver, gold
    "reservations": ["ABC123", "XYZ789"]
}
"""

AIRLINE_RESERVATION_SCHEMA = """
{
    "reservation_id": "ABC123",
    "user_id": "john_doe_1234",
    "origin": "JFK",
    "destination": "LAX",
    "flight_type": "round_trip",  // Can be: one_way, round_trip
    "cabin": "economy",  // Can be: basic_economy, economy, business
    "flights": [
        {
            "origin": "JFK",
            "destination": "LAX",
            "flight_number": "HAT100",
            "date": "2024-05-20",
            "price": 350
        },
        {
            "origin": "LAX",
            "destination": "JFK",
            "flight_number": "HAT101",
            "date": "2024-05-25",
            "price": 380
        }
    ],
    "passengers": [
        { "first_name": "John", "last_name": "Doe", "dob": "1985-06-15" }
    ],
    "payment_history": [
        { "payment_id": "credit_card_1234567", "amount": 730 }
    ],
    "created_at": "2024-05-01T10:30:00",
    "total_baggages": 2,
    "nonfree_baggages": 1,
    "insurance": "yes"  // Can be: yes, no
}
"""

AIRLINE_AIRPORTS = """
Available airports: JFK, LAX, ORD, DFW, DEN, SFO, SEA, ATL, BOS, MIA, 
PHX, IAH, LAS, MCO, EWR, MSP, DTW, PHL, LGA, CLT
"""


class PersonaCreator:
    """Creates custom personas with generated data for simulation."""

    def __init__(self, env_name: str, model: str, provider: str):
        self.env_name = env_name
        self.model = model
        self.provider = provider
        self.env_config: Optional[EnvironmentConfig] = None
        self._task_examples: Optional[List[str]] = None
        
        # Load environment config from registry
        try:
            self.env_config = get_environment_config(env_name)
        except ValueError:
            console.print(f"[yellow]Warning: Environment '{env_name}' not found in registry. Using fallback schemas.[/yellow]")
        
        # Load task examples from the environment
        self._load_task_examples()
        
        ensure_personas_dir()
    
    def _load_task_examples(self) -> None:
        """Load task instructions from the environment's tasks.py file."""
        try:
            # Dynamically import the tasks module for the environment
            module_name = f"sigma.envs.{self.env_name}.tasks"
            tasks_module = importlib.import_module(module_name)
            
            if hasattr(tasks_module, 'tasks'):
                tasks = tasks_module.tasks
                # Extract unique instructions (some tasks have similar instructions)
                instructions = list(set(task.get('instruction', '') for task in tasks if task.get('instruction')))
                self._task_examples = instructions
                console.print(f"[dim]Loaded {len(self._task_examples)} task examples from {self.env_name} environment[/dim]")
            else:
                console.print(f"[yellow]Warning: No tasks found in {module_name}[/yellow]")
                self._task_examples = []
        except ImportError as e:
            console.print(f"[yellow]Warning: Could not load tasks for '{self.env_name}': {e}[/yellow]")
            self._task_examples = []
    
    def _get_sample_task_instructions(self, num_samples: int = 10) -> str:
        """Get a sample of task instructions to use as examples."""
        if not self._task_examples:
            return ""
        
        # Sample random instructions, or all if fewer than requested
        sample_size = min(num_samples, len(self._task_examples))
        sampled = random.sample(self._task_examples, sample_size)
        
        examples = "\n\n".join(f"Example {i+1}:\n{instr}" for i, instr in enumerate(sampled))
        return examples

    def _get_schema_context(self) -> str:
        """Get the appropriate schema context for the environment."""
        # Try to use registry-provided schemas if available
        if self.env_config and self.env_config.persona_schemas:
            return self.env_config.persona_schemas
        
        # Fallback to hardcoded schemas for known environments
        if self.env_name == "retail":
            return f"""
# User Schema
{RETAIL_USER_SCHEMA}

# Order Schema
{RETAIL_ORDER_SCHEMA}

# Available Products
{RETAIL_PRODUCTS_INFO}
"""
        elif self.env_name == "airline":
            return f"""
# User Schema
{AIRLINE_USER_SCHEMA}

# Reservation Schema
{AIRLINE_RESERVATION_SCHEMA}

# Available Airports
{AIRLINE_AIRPORTS}
"""
        else:
            # For unknown environments, try to build schema from data
            console.print(f"[yellow]Warning: No schema defined for '{self.env_name}'. Generation may be limited.[/yellow]")
            return "No specific schema available. Generate reasonable test data."
    
    def _get_scenario_examples(self) -> List[str]:
        """Get scenario examples for the environment - uses actual task instructions."""
        # If we have loaded task examples, extract simplified scenario descriptions
        if self._task_examples and len(self._task_examples) > 0:
            # Sample a few task instructions and simplify them for display
            sample_size = min(5, len(self._task_examples))
            sampled = random.sample(self._task_examples, sample_size)
            
            # Extract short scenario descriptions from the full instructions
            simplified = []
            for instr in sampled:
                # Take just the core task, truncate if too long
                # Skip the "You are X" part and get the main action
                parts = instr.split('. ')
                if len(parts) > 1:
                    # Get the main task part (usually after the "You are X" intro)
                    task_part = '. '.join(parts[1:3])  # Take 2nd and 3rd sentences
                    if len(task_part) > 120:
                        task_part = task_part[:120] + "..."
                    simplified.append(task_part)
                else:
                    simplified.append(instr[:120] + "..." if len(instr) > 120 else instr)
            
            return simplified
        
        # Fallback if no task examples loaded
        if self.env_config and self.env_config.scenario_examples:
            return self.env_config.scenario_examples
        
        # Default fallback examples
        if self.env_name == "retail":
            return [
                "Customer wants to return a delivered laptop because it's defective",
                "Customer wants to cancel a pending order and use a gift card for refund",
                "Customer wants to exchange a t-shirt for a different size"
            ]
        elif self.env_name == "airline":
            return [
                "Customer wants to change flight to an earlier time",
                "Customer wants to cancel reservation and get refund",
                "Customer wants to add extra baggage to booking"
            ]
        else:
            return [
                "Customer has a typical support request",
                "Customer wants to modify their account or order"
            ]
    
    def _get_data_key(self) -> str:
        """Get the key name for order/reservation data in the environment."""
        if self.env_config and self.env_config.data_key:
            return self.env_config.data_key
        
        # Fallback based on known environments
        if self.env_name == "retail":
            return "orders"
        elif self.env_name == "airline":
            return "reservations"
        else:
            return "data"

    def generate_persona_data(self, scenario: str) -> Dict[str, Any]:
        """Use LLM to generate persona data based on scenario."""
        schema_context = self._get_schema_context()
        task_examples = self._get_sample_task_instructions(num_samples=8)
        
        if self.env_name == "retail":
            prompt = f"""You are generating test data for a retail customer service simulation.

{schema_context}

SCENARIO: {scenario}

=== REFERENCE TASK INSTRUCTION EXAMPLES ===
Below are examples of persona instructions used in this simulation. Your generated instruction should follow a similar style and focus ONLY on realistic customer service scenarios (returns, exchanges, order modifications, cancellations, product inquiries, etc.). Do NOT include unrelated activities like guessing poems, trivia, or anything not related to retail customer service.

{task_examples}

=== END OF EXAMPLES ===

Generate realistic data for this scenario. Create:
1. A user profile with appropriate payment methods
2. One or more orders that fit the scenario (with realistic items, prices, and statuses)

IMPORTANT RULES:
- Generate valid JSON only
- Order IDs must be in format "#W" followed by 7 digits
- Item IDs and product IDs should be 10-digit numbers as strings
- Payment method IDs should follow the format shown in schema
- User ID should be firstname_lastname_XXXX format
- Make sure the order status matches what the scenario needs (e.g., "delivered" for returns/exchanges, "pending" for cancellations)
- Include realistic prices (items typically $20-500)
- The instruction MUST focus ONLY on retail customer service tasks (returns, exchanges, modifications, cancellations, product questions)
- Include personality traits like "detail-oriented", "private person", "reactive to the agent", etc. as shown in examples
- Do NOT include any off-topic requests (like poems, trivia, games, etc.)

Return a JSON object with this structure:
{{
    "user": {{ ... user object ... }},
    "orders": {{ 
        "#W1234567": {{ ... order object ... }},
        ...
    }},
    "instruction": "A detailed persona instruction for the user simulator describing what they want to accomplish and their personality - MUST be retail customer service focused like the examples above"
}}
"""
        else:  # airline
            prompt = f"""You are generating test data for an airline customer service simulation.

{schema_context}

The current simulation date is 2024-05-15.

SCENARIO: {scenario}

=== REFERENCE TASK INSTRUCTION EXAMPLES ===
Below are examples of persona instructions used in this simulation. Your generated instruction should follow a similar style and focus ONLY on realistic airline customer service scenarios (flight changes, cancellations, baggage, passengers, payments, etc.). Do NOT include unrelated activities like guessing poems, trivia, or anything not related to airline customer service.

{task_examples}

=== END OF EXAMPLES ===

Generate realistic data for this scenario. Create:
1. A user profile with appropriate payment methods and membership
2. One or more reservations that fit the scenario

IMPORTANT RULES:
- Generate valid JSON only
- Reservation IDs should be 6 alphanumeric characters (uppercase)
- Flight numbers should be "HAT" followed by 3 digits (e.g., HAT123)
- Use realistic airport codes from the list
- Use dates around May 2024 (simulation date is 2024-05-15)
- User ID should be firstname_lastname_XXXX format
- Make sure flight dates and statuses match the scenario needs
- Prices: basic_economy ~$80-150, economy ~$150-300, business ~$400-900 per segment
- The instruction MUST focus ONLY on airline customer service tasks (flight changes, cancellations, baggage, payments, passenger updates)
- Include personality traits like "reactive to the agent", "calm", "emotional", etc. as shown in examples
- Do NOT include any off-topic requests (like poems, trivia, games, etc.)

Return a JSON object with this structure:
{{
    "user": {{ ... user object ... }},
    "reservations": {{ 
        "ABC123": {{ ... reservation object ... }},
        ...
    }},
    "instruction": "A detailed persona instruction for the user simulator describing what they want to accomplish and their personality - MUST be airline customer service focused like the examples above"
}}
"""

        console.print("[bold]Generating persona data...[/bold]")
        with console.status("[bold green]Calling LLM...[/bold green]"):
            try:
                res = completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                response_text = res.choices[0].message.content
                
                # Extract JSON from response
                # Try to find JSON block
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end]
                
                data = json.loads(response_text.strip())
                return data
                
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing JSON: {e}[/red]")
                console.print(f"[dim]Response was: {response_text[:500]}...[/dim]")
                raise
            except Exception as e:
                console.print(f"[red]Error generating data: {e}[/red]")
                raise

    def display_generated_data(self, data: Dict[str, Any]):
        """Display the generated persona data."""
        console.print("\n")
        console.print(Panel(
            data.get("instruction", "No instruction generated"),
            title="[bold yellow]Generated Persona Instruction[/bold yellow]",
            border_style="yellow"
        ))
        
        console.print("\n[bold cyan]Generated User Profile:[/bold cyan]")
        user_json = json.dumps(data.get("user", {}), indent=2)
        console.print(Syntax(user_json, "json", theme="monokai"))
        
        # Display environment-specific data
        data_key = self._get_data_key()
        items = data.get(data_key, {})
        
        if items:
            console.print(f"\n[bold cyan]Generated {data_key.title()} ({len(items)}):[/bold cyan]")
            for item_id, item in items.items():
                # Try to build a descriptive summary
                summary_parts = []
                if "status" in item:
                    summary_parts.append(f"Status: {item.get('status')}")
                if "origin" in item and "destination" in item:
                    summary_parts.append(f"{item.get('origin')} → {item.get('destination')}")
                
                summary = " - ".join(summary_parts) if summary_parts else ""
                console.print(f"\n[green]{item_id}[/green]{' - ' + summary if summary else ''}")
                item_json = json.dumps(item, indent=2)
                console.print(Syntax(item_json, "json", theme="monokai"))

    def save_persona(self, name: str, data: Dict[str, Any]) -> str:
        """Save the persona to a file."""
        filename = f"{name}_{self.env_name}.json"
        filepath = os.path.join(PERSONAS_DIR, filename)
        
        data_key = self._get_data_key()
        
        persona_data = {
            "name": name,
            "env": self.env_name,
            "created_at": datetime.now().isoformat(),
            "instruction": data.get("instruction", ""),
            "user": data.get("user", {}),
            data_key: data.get(data_key, {}),
        }
        
        with open(filepath, "w") as f:
            json.dump(persona_data, f, indent=2)
        
        return filepath

    def run(self):
        """Run the interactive persona creation session."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold blue]🎭 Persona Creator[/bold blue]\n\n"
            f"Environment: [cyan]{self.env_name}[/cyan]\n"
            "Create custom personas with generated data for simulation.",
            title="Welcome",
            border_style="blue"
        ))
        
        # Get scenario from user
        console.print("\n[bold]Describe the scenario you want to simulate:[/bold]")
        console.print("[dim]Examples:[/dim]")
        
        # Use dynamic examples from registry or fallback
        for example in self._get_scenario_examples():
            console.print(f"[dim]- {example}[/dim]")
        
        console.print()
        scenario = Prompt.ask("[green]Your scenario[/green]")
        
        if not scenario.strip():
            console.print("[red]No scenario provided. Exiting.[/red]")
            return
        
        # Generate data
        try:
            data = self.generate_persona_data(scenario)
        except Exception as e:
            console.print(f"[red]Failed to generate persona: {e}[/red]")
            return
        
        # Display generated data
        self.display_generated_data(data)
        
        # Ask for confirmation
        console.print()
        if not Confirm.ask("Save this persona?", default=True):
            console.print("[yellow]Persona not saved.[/yellow]")
            return
        
        # Get persona name
        persona_name = Prompt.ask(
            "[green]Enter a name for this persona[/green]",
            default=f"persona_{generate_id(4)}"
        )
        
        # Save persona
        filepath = self.save_persona(persona_name, data)
        console.print(f"\n[green]✓ Persona saved to:[/green] {filepath}")
        
        # Show how to use it
        console.print("\n[bold]To use this persona in simulation:[/bold]")
        console.print(f"  python -m sigma --env {self.env_name} --persona-file {filepath}")


def list_personas(env_name: Optional[str] = None):
    """List all saved personas."""
    ensure_personas_dir()
    
    personas = []
    for filename in os.listdir(PERSONAS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(PERSONAS_DIR, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                if env_name is None or data.get("env") == env_name:
                    personas.append({
                        "filename": filename,
                        "filepath": filepath,
                        "name": data.get("name", "Unknown"),
                        "env": data.get("env", "Unknown"),
                        "created_at": data.get("created_at", "Unknown"),
                        "instruction": data.get("instruction", "")[:80] + "..."
                    })
            except Exception:
                pass
    
    if not personas:
        console.print("[yellow]No personas found.[/yellow]")
        return
    
    table = Table(title="Saved Personas")
    table.add_column("Name", style="green")
    table.add_column("Environment", style="cyan")
    table.add_column("Created", style="white")
    table.add_column("Instruction", style="dim", max_width=50)
    
    for p in personas:
        table.add_row(
            p["name"],
            p["env"],
            p["created_at"][:10] if p["created_at"] != "Unknown" else "Unknown",
            p["instruction"]
        )
    
    console.print(table)


def main():
    # Get available environments from registry
    available_envs = list_environments()
    
    parser = argparse.ArgumentParser(
        description="Create custom personas for tau-bench simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Create a persona for retail environment
  python -m sigma.persona_creator --env retail --model gpt-4o --provider openai

  # List all saved personas
  python -m sigma.persona_creator --list

  # List personas for a specific environment
  python -m sigma.persona_creator --list --env airline

Available environments: {', '.join(available_envs)}
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        choices=available_envs,
        default=available_envs[0] if available_envs else "retail",
        help=f"Environment to create persona for (default: {available_envs[0] if available_envs else 'retail'})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model for data generation (default: gpt-4o)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all saved personas"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_personas(args.env if args.env else None)
        return
    
    creator = PersonaCreator(
        env_name=args.env,
        model=args.model,
        provider=args.provider,
    )
    
    try:
        creator.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
