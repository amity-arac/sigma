# Copyright Amity
"""
Scenario Generator for tau-bench Simulator

This module generates new, unique scenarios inspired by existing tasks.
It creates complete persona data including user profiles, orders/reservations,
and detailed instructions that can be used to mock data in the environment.

The generator is ENVIRONMENT-INDEPENDENT:
1. Sends raw DB structure as JSON to the LLM
2. LLM generates scenario AND all required data together
3. Generated data is augmented into the DB during simulation

Usage:
    from sigma.scenario_generator import generate_scenario_for_env
    scenario = generate_scenario_for_env("retail", "gpt-4o", "openai")
"""

import importlib
import json
import random
import string
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from litellm import completion

from sigma.env_registry import get_environment_config, EnvironmentConfig, DATA_ENVS_PATH


@dataclass
class GeneratedScenario:
    """A generated scenario with all required data."""
    instruction: str
    user: Dict[str, Any]
    data: Dict[str, Any]  # orders/reservations/etc based on env
    data_key: str  # "orders", "reservations", etc.
    seed_task_instruction: str  # The original task that inspired this scenario
    generation_timestamp: str
    env_name: str
    user_id: str = ""  # Generated user_id for GRPO training data
    scenario_goal: str = ""  # Overall expected outcome of this scenario
    augmented_data: Dict[str, Any] = field(default_factory=dict)  # Additional data to inject (products, flights, etc.)
    seed_task_id: Optional[str] = None  # The ID of the original task that inspired this scenario
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instruction": self.instruction,
            "user": self.user,
            "user_id": self.user_id,
            "data": self.data,
            "data_key": self.data_key,
            "seed_task_instruction": self.seed_task_instruction,
            "seed_task_id": self.seed_task_id,
            "generation_timestamp": self.generation_timestamp,
            "env_name": self.env_name,
            "scenario_goal": self.scenario_goal,
            "augmented_data": self.augmented_data,
        }


class ScenarioGenerator:
    """
    Generates new simulation scenarios inspired by existing tasks.
    
    This generator is ENVIRONMENT-INDEPENDENT:
    - Sends raw DB structure as JSON to the LLM  
    - LLM generates scenario AND all required data together
    - Generated data is augmented into the DB during simulation
    
    Creates complete, runnable scenarios with:
    - New user profiles (different names, addresses, payment methods)
    - New data records (orders/reservations) with realistic content
    - Any additional data needed (products, flights, etc.)
    - Detailed persona instructions with personality traits
    """
    
    def __init__(
        self,
        env_name: str,
        model: str = "gpt-5.2",
        provider: str = "openai",
        task_ids: Optional[List[int]] = None,
        num_db_samples: int = 3,  # Number of sample records to show LLM from each collection
    ):
        self.env_name = env_name
        self.model = model
        self.provider = provider
        self.task_ids = task_ids  # Optional list of task IDs to sample from
        self.num_db_samples = num_db_samples
        
        # Load environment config from registry (required)
        self.env_config: EnvironmentConfig = get_environment_config(env_name)
        
        # Load tasks, data, and policy
        self._tasks: List[Dict[str, Any]] = []
        self._env_data: Dict[str, Any] = {}
        self._policy: str = ""
        self._load_tasks()
        self._load_env_data()
        self._load_policy()
    
    def _load_policy(self) -> None:
        """Load the agent policy document for this environment."""
        import os
        # Use DATA_ENVS_PATH to find policy.md in data/envs/<env_name>/
        policy_path = os.path.join(DATA_ENVS_PATH, self.env_name, "policy.md")
        
        # Also check for retail_t2 -> retail fallback
        if not os.path.exists(policy_path) and self.env_name == "retail_t2":
            policy_path = os.path.join(DATA_ENVS_PATH, "retail", "policy.md")
        
        try:
            if os.path.exists(policy_path):
                with open(policy_path, "r") as f:
                    self._policy = f.read()
                print(f"[ScenarioGenerator] Loaded policy from {policy_path}")
            else:
                print(f"[ScenarioGenerator] Warning: No policy.md found at {policy_path}")
                self._policy = ""
        except Exception as e:
            print(f"[ScenarioGenerator] Warning: Could not load policy: {e}")
            self._policy = ""
    
    def _load_tasks(self) -> None:
        """Load tasks from the environment's task loader."""
        try:
            if self.env_config.tasks_loader:
                tasks = self.env_config.tasks_loader("test")
                # Convert to dict format if needed, handling both new and legacy formats
                self._tasks = []
                for task in tasks:
                    if isinstance(task, dict):
                        # Check if it's new retail_t2 format (has user_scenario)
                        if "user_scenario" in task:
                            # New format - extract instruction
                            from sigma.envs.generic_env import get_task_instruction
                            self._tasks.append({
                                "id": task.get("id"),
                                "instruction": get_task_instruction(task),
                                "user_scenario": task.get("user_scenario"),
                                "evaluation_criteria": task.get("evaluation_criteria"),
                            })
                        else:
                            # Legacy format
                            self._tasks.append(task)
                    else:
                        # Object with attributes (legacy Task objects)
                        self._tasks.append({
                            "instruction": task.instruction,
                            "user_id": task.user_id,
                            "actions": task.actions,
                        })
        except Exception as e:
            print(f"[ScenarioGenerator] Warning: Could not load tasks: {e}")
            self._tasks = []
    
    def _load_env_data(self) -> None:
        """Load environment data (products, flights, etc.) from data loader."""
        try:
            if self.env_config.data_loader:
                self._env_data = self.env_config.data_loader()
        except Exception as e:
            print(f"[ScenarioGenerator] Warning: Could not load env data: {e}")
            self._env_data = {}
    
    def _sample_seed_tasks(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Sample random tasks to use as inspiration.
        
        If task_ids is set, only sample from tasks with those IDs.
        """
        if not self._tasks:
            return []
        
        # Filter by task IDs if specified
        available_tasks = self._tasks
        if self.task_ids is not None:
            # Convert task_ids to strings for comparison since task IDs in JSON are strings
            task_ids_str = [str(tid) for tid in self.task_ids]
            available_tasks = [
                task for task in self._tasks 
                if str(task.get("id")) in task_ids_str
            ]
            if not available_tasks:
                print(f"[ScenarioGenerator] Warning: No tasks found with IDs {self.task_ids}, using all tasks")
                available_tasks = self._tasks
            else:
                print(f"[ScenarioGenerator] Filtering to {len(available_tasks)} tasks with IDs: {self.task_ids}")
        
        sample_size = min(num_samples, len(available_tasks))
        return random.sample(available_tasks, sample_size)
    
    def _sample_db_structure(self) -> Dict[str, Any]:
        """
        Sample the database structure with example records.
        This gives the LLM real examples of what data looks like.
        
        Returns a dict with sampled records from each collection.
        """
        sampled = {}
        
        for collection_name, collection_data in self._env_data.items():
            if not isinstance(collection_data, dict) or not collection_data:
                continue
            
            # Sample a few records from each collection
            all_ids = list(collection_data.keys())
            sample_ids = random.sample(all_ids, min(self.num_db_samples, len(all_ids)))
            
            sampled[collection_name] = {
                "_total_count": len(collection_data),
                "_sample_records": {
                    record_id: collection_data[record_id]
                    for record_id in sample_ids
                }
            }
        
        return sampled
    
    def _generate_base_ids(self) -> Dict[str, str]:
        """Generate basic unique IDs that work across environments."""
        first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
            "Isabella", "Lucas", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
            "Jacob", "Harper", "Michael", "Evelyn", "Alexander", "Yuki", "Chen",
            "Fatima", "Omar", "Priya", "Carlos", "Aisha", "Dmitri", "Kenji", "Rosa"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Kim", "Singh", "Patel", "Chen", "Wang", "Ali", "Khan", "Tanaka"
        ]
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        random_suffix = ''.join(random.choices(string.digits, k=4))
        
        return {
            "first_name": first_name,
            "last_name": last_name,
            "user_id": f"{first_name.lower()}_{last_name.lower()}_{random_suffix}",
            "zip": f"{random.randint(10000, 99999)}",
            "credit_card_id": f"credit_card_{random.randint(1000000, 9999999)}",
            "gift_card_id": f"gift_card_{random.randint(1000000, 9999999)}",
            "last_four": f"{random.randint(1000, 9999)}",
        }
    
    def _build_prompt(self, seed_tasks: List[Dict[str, Any]], ids: Dict[str, str], db_samples: Dict[str, Any]) -> str:
        """
        Build a prompt with chain-of-thought reasoning.
        
        Forces the LLM to think step-by-step:
        1. Design the scenario concept
        2. Define expected goals
        3. Work backwards to determine required data
        """
        
        # Get seed task instructions
        seed_instructions = "\n\n".join([
            f"Example {i+1}:\n{task.get('instruction', '')}"
            for i, task in enumerate(seed_tasks)
        ])
        
        # Get data key from registry
        data_key = self.env_config.data_key
        
        # Format DB samples as JSON for the LLM
        db_json = json.dumps(db_samples, indent=2, default=str)
        
        # Include policy section if available
        policy_section = ""
        if self._policy:
            policy_section = f"""
=== AGENT POLICY (What the agent is supposed to do) ===
This is the policy document that guides what the agent can and cannot do.
Use this to understand what scenarios are valid and what actions are available.

{self._policy}

"""
        
        return f"""You are generating a NEW, UNIQUE test scenario for a {self.env_config.display_name} customer service simulation.

=== ENVIRONMENT: {self.env_config.display_name} ===
{self.env_config.description}
{policy_section}
=== DATABASE STRUCTURE (Sample Records from Each Collection) ===
Study this structure carefully. Your generated data MUST follow these exact schemas.

```json
{db_json}
```

=== EXISTING TASK EXAMPLES (Use as INSPIRATION for the TYPE of scenarios) ===
{seed_instructions}

=== REQUIRED IDS TO USE ===
- User ID: {ids['user_id']}
- First Name: {ids['first_name']}
- Last Name: {ids['last_name']}
- Zip Code: {ids['zip']}
- Credit Card ID: {ids['credit_card_id']}
- Credit Card Last Four: {ids['last_four']}

=== CHAIN OF THOUGHT: THINK STEP BY STEP ===

You MUST reason through this step-by-step before generating the final JSON.

**STEP 1: DESIGN THE SCENARIO CONCEPT**
Think about:
- What kind of customer service situation will this be? (return, exchange, cancellation, inquiry, etc.)
- What personality will the customer have? (patient, impatient, confused, demanding, etc.)
- What makes this scenario interesting or challenging?

**STEP 2: DEFINE THE EXPECTED OUTCOME (GOAL)**
Think about:
- What should the agent DO to successfully handle this?
- What tool calls should the agent make?
- What information should be communicated to the customer?
- What is the end state after successful resolution?

**STEP 3: WORK BACKWARDS - WHAT DATA IS NEEDED?**
Based on Steps 1 and 2, determine:
- What product(s) does the customer need to have ordered?
- What are the EXACT specifications of those products (variants, options, prices)?
- If it's an exchange scenario: What product variant does the customer CURRENTLY have? What do they want to exchange TO?
- BOTH the current item AND target item MUST be defined with complete, matching data

**STEP 4: CREATE THE DATA WITH FULL CONSISTENCY**
Now create all data ensuring:
- Every item_id in an order references a real product variant you define
- Product variants have complete options matching what the order says
- Prices, names, and options are consistent across all references
- The user's order status is appropriate (pending, delivered, etc.)

=== OUTPUT FORMAT ===

First, write your reasoning in a <thinking> block, then output the JSON.

<thinking>
STEP 1 - Scenario Concept:
[Your reasoning about what scenario to create...]

STEP 2 - Expected Goal:
[Your reasoning about what the successful outcome looks like...]

STEP 3 - Required Data:
[Your reasoning about what specific data items are needed...]
- Product needed: [name] with variants: [list specific variants with their options]
- Customer currently has: [specific variant with item_id]
- Customer wants: [if exchange, specific target variant with item_id]

STEP 4 - Data Consistency Check:
[Verify all IDs match, all options are consistent...]
</thinking>

Then output ONLY the JSON object:
```json
{{
    "instruction": "You are {ids['first_name']} {ids['last_name']} (user id: {ids['user_id']}). [Describe what they want and their personality]. [Any specific conditions or preferences].",
    "scenario_goal": "[Clear description of: 1) What actions the agent should take, 2) What the end state should be]",
    "user": {{
        "user_id": "{ids['user_id']}",
        "name": {{ "first_name": "{ids['first_name']}", "last_name": "{ids['last_name']}" }},
        ... complete user profile following DB structure ...
    }},
    "{data_key}": {{
        "#W[order_id]": {{
            ... order/reservation following DB structure exactly ...
            "items": [
                {{
                    "name": "[product name]",
                    "product_id": "[must match a product in augmented_data]",
                    "item_id": "[must match a variant in that product]",
                    "options": {{ ... must match the variant's options exactly ... }},
                    "price": [must match the variant's price]
                }}
            ]
        }}
    }},
    "augmented_data": {{
        "products": {{
            "[product_id]": {{
                "name": "[product name]",
                "product_id": "[same as key]",
                "variants": {{
                    "[item_id_1]": {{
                        "item_id": "[same as key]",
                        "options": {{ ... all option key-values ... }},
                        "available": true,
                        "price": [price as number]
                    }},
                    "[item_id_2]": {{
                        ... another variant if needed for exchange ...
                    }}
                }}
            }}
        }}
    }}
}}
```

CRITICAL REMINDERS:
- item_id in order items MUST exist as a key in the product's variants
- options in order items MUST exactly match the variant's options  
- price in order items MUST exactly match the variant's price
- For exchanges: BOTH source and target variants must exist in the same product
"""
    
    def generate(self) -> GeneratedScenario:
        """
        Generate a new scenario with all required data.
        
        The LLM generates both the scenario AND any data needed,
        which will be augmented into the DB during simulation.
        
        Returns:
            GeneratedScenario with all data needed to run a simulation
        """
        # Sample seed tasks for inspiration
        seed_tasks = self._sample_seed_tasks(5)
        if not seed_tasks:
            raise ValueError(f"No tasks found for environment '{self.env_name}'")
        
        # Pick one as the primary inspiration
        primary_seed = random.choice(seed_tasks)
        seed_instruction = primary_seed.get("instruction", "")
        seed_task_id = primary_seed.get("id")  # Get the task ID
        
        # Generate unique IDs
        ids = self._generate_base_ids()
        
        # Sample DB structure to show the LLM
        db_samples = self._sample_db_structure()
        
        # Build the prompt
        prompt = self._build_prompt(seed_tasks, ids, db_samples)
        
        # Log generation start
        print(f"\n{'='*70}")
        print("🎲 GENERATING NEW SCENARIO")
        print(f"{'='*70}")
        print(f"🌍 Environment: {self.env_config.display_name}")
        print(f"🤖 Model: {self.model} ({self.provider})")
        print(f"📌 Seed task ID: {seed_task_id}")
        print(f"📌 Seed inspiration: {seed_instruction[:80]}...")
        print(f"📊 DB collections sampled: {', '.join(db_samples.keys())}")
        print(f"{'='*70}\n")
        
        try:
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = res.choices[0].message.content
            
            # Parse JSON from response
            data = self._parse_llm_response(response_text)
            
            # Get data key from registry
            data_key = self.env_config.data_key
            
            # Extract user_id from generated user data
            user_data = data.get("user", {})
            user_id = user_data.get("user_id", "")
            if not user_id:
                # Try to generate user_id from name if not provided
                name = user_data.get("name", {})
                if isinstance(name, dict):
                    first_name = name.get("first_name", "user")
                    last_name = name.get("last_name", "generated")
                    user_id = f"{first_name.lower()}_{last_name.lower()}_{random.randint(1000, 9999)}"
                elif isinstance(name, str):
                    user_id = f"{name.lower().replace(' ', '_')}_{random.randint(1000, 9999)}"
                else:
                    user_id = f"generated_user_{random.randint(1000, 9999)}"
            
            # Create the scenario
            scenario = GeneratedScenario(
                instruction=data.get("instruction", ""),
                user=user_data,
                user_id=user_id,
                data=data.get(data_key, {}),
                data_key=data_key,
                seed_task_instruction=seed_instruction,
                seed_task_id=seed_task_id,
                generation_timestamp=datetime.now().isoformat(),
                env_name=self.env_name,
                scenario_goal=data.get("scenario_goal", ""),
                augmented_data=data.get("augmented_data", {}),
            )
            
            return scenario
            
        except Exception as e:
            print(f"❌ Error generating scenario: {e}")
            raise
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        
        Handles chain-of-thought responses by:
        1. Extracting and logging the <thinking> block
        2. Finding and parsing the JSON block
        """
        # Extract and log thinking block if present
        if "<thinking>" in response_text:
            thinking_start = response_text.find("<thinking>") + len("<thinking>")
            thinking_end = response_text.find("</thinking>")
            if thinking_end > thinking_start:
                thinking = response_text[thinking_start:thinking_end].strip()
                print(f"\n{'─'*70}")
                print("🧠 LLM REASONING (Chain of Thought)")
                print(f"{'─'*70}")
                # Print first 500 chars of thinking
                if len(thinking) > 500:
                    print(f"{thinking[:500]}...")
                else:
                    print(thinking)
                print(f"{'─'*70}\n")
        
        # Try to extract JSON block
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end]
        else:
            # No code block, try to find JSON object directly
            # Look for the outermost { }
            brace_start = response_text.find("{")
            if brace_start != -1:
                # Find matching closing brace
                depth = 0
                for i, char in enumerate(response_text[brace_start:], brace_start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            response_text = response_text[brace_start:i+1]
                            break
        
        return json.loads(response_text.strip())

    def get_full_prompt(self, num_seed_tasks: int = 5) -> str:
        """
        Generate and return the complete prompt that would be sent to the LLM.
        
        This method builds the full prompt without sending it to the LLM,
        useful for debugging and inspection.
        
        Args:
            num_seed_tasks: Number of seed tasks to sample (default 5)
            
        Returns:
            The complete prompt string as it would be sent to the LLM
        """
        # Sample seed tasks for inspiration
        seed_tasks = self._sample_seed_tasks(num_seed_tasks)
        if not seed_tasks:
            raise ValueError(f"No tasks found for environment '{self.env_name}'")
        
        # Generate unique IDs
        ids = self._generate_base_ids()
        
        # Sample DB structure to show the LLM
        db_samples = self._sample_db_structure()
        
        # Build and return the prompt
        return self._build_prompt(seed_tasks, ids, db_samples)
    
    def print_full_prompt(self, num_seed_tasks: int = 5) -> None:
        """
        Print the complete prompt to console without truncation.
        
        Args:
            num_seed_tasks: Number of seed tasks to sample (default 5)
        """
        prompt = self.get_full_prompt(num_seed_tasks)
        print(f"\n{'='*80}")
        print("📝 FULL SCENARIO GENERATOR PROMPT")
        print(f"{'='*80}")
        print(f"Environment: {self.env_name}")
        print(f"Model: {self.model}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"{'='*80}\n")
        print(prompt)
        print(f"\n{'='*80}")
        print("📝 END OF PROMPT")
        print(f"{'='*80}\n")


def log_scenario_to_console(scenario: GeneratedScenario) -> None:
    """
    Log the generated scenario to console in a readable format.
    This should be called BEFORE starting the simulation.
    """
    print(f"\n{'='*70}")
    print("✅ GENERATED SCENARIO READY")
    print(f"{'='*70}")
    
    print(f"\n🌍 ENVIRONMENT: {scenario.env_name.upper()}")
    print(f"⏰ Generated: {scenario.generation_timestamp}")
    
    print(f"\n{'─'*70}")
    print("📋 PERSONA INSTRUCTION")
    print(f"{'─'*70}")
    print(f"{scenario.instruction}")
    
    print(f"\n{'─'*70}")
    print("👤 USER PROFILE")
    print(f"{'─'*70}")
    user = scenario.user
    name = user.get("name", {})
    print(f"  Name:     {name.get('first_name', '')} {name.get('last_name', '')}")
    print(f"  User ID:  {user.get('user_id', '')}")
    
    if user.get("email"):
        print(f"  Email:    {user.get('email')}")
    
    address = user.get("address", {})
    if address:
        addr_parts = [address.get('address1', '')]
        if address.get('address2'):
            addr_parts.append(address.get('address2'))
        addr_parts.extend([
            address.get('city', ''),
            address.get('state', ''),
            address.get('zip', '')
        ])
        print(f"  Address:  {', '.join(p for p in addr_parts if p)}")
    
    if user.get("dob"):
        print(f"  DOB:      {user.get('dob')}")
    
    if user.get("membership"):
        print(f"  Member:   {user.get('membership')}")
    
    # Payment methods
    payment_methods = user.get("payment_methods", {})
    if payment_methods:
        print(f"  Payments:")
        for pm_id, pm in payment_methods.items():
            source = pm.get("source", "unknown")
            if source == "credit_card":
                print(f"    - {pm_id}: Credit card ****{pm.get('last_four', '****')}")
            elif source == "gift_card":
                print(f"    - {pm_id}: Gift card (${pm.get('balance', pm.get('amount', 0))})")
            elif source == "certificate":
                print(f"    - {pm_id}: Certificate (${pm.get('amount', 0)})")
            elif source == "paypal":
                print(f"    - {pm_id}: PayPal")
            else:
                print(f"    - {pm_id}: {source}")
    
    print(f"\n{'─'*70}")
    print(f"📦 {scenario.data_key.upper()} DATA")
    print(f"{'─'*70}")
    
    for record_id, record in scenario.data.items():
        print(f"\n  [{record_id}]")
        
        # Common fields
        if "status" in record:
            print(f"    Status: {record.get('status')}")
        
        # Airline-specific
        if "origin" in record and "destination" in record:
            print(f"    Route: {record.get('origin')} → {record.get('destination')}")
        if "cabin" in record:
            print(f"    Cabin: {record.get('cabin')}")
        if "flight_type" in record:
            print(f"    Type: {record.get('flight_type')}")
        if "flights" in record:
            flights = record.get("flights", [])
            print(f"    Flights: {len(flights)} segment(s)")
            for f in flights[:3]:  # Show first 3
                if isinstance(f, dict):
                    print(f"      - {f.get('flight_number', 'N/A')} on {f.get('date', 'N/A')}")
        if "passengers" in record:
            passengers = record.get("passengers", [])
            print(f"    Passengers: {len(passengers)}")
        if "total_baggages" in record:
            print(f"    Bags: {record.get('total_baggages')}")
        if "insurance" in record:
            print(f"    Insurance: {record.get('insurance')}")
        
        # Retail-specific
        if "items" in record and "flights" not in record:
            items = record.get("items", [])
            print(f"    Items ({len(items)}):")
            for item in items[:5]:  # Show first 5
                name = item.get("name", "Unknown")
                price = item.get("price", 0)
                print(f"      - {name}: ${price:.2f}")
                options = item.get("options", {})
                if options:
                    opt_str = ", ".join(f"{k}={v}" for k, v in list(options.items())[:3])
                    print(f"        ({opt_str})")
    
    # Show augmented data if present
    if scenario.augmented_data:
        print(f"\n{'─'*70}")
        print("📦 AUGMENTED DATA (Will be injected into DB)")
        print(f"{'─'*70}")
        for collection_name, collection_data in scenario.augmented_data.items():
            if collection_data:
                print(f"  {collection_name}: {len(collection_data)} record(s)")
                for record_id in list(collection_data.keys())[:3]:
                    print(f"    - {record_id}")
    
    print(f"\n{'─'*70}")
    print("🎯 EXPECTED OUTCOME (SCENARIO GOAL)")
    print(f"{'─'*70}")
    if scenario.scenario_goal:
        print(f"  {scenario.scenario_goal}")
    else:
        print("  (No explicit goal specified)")
    
    print(f"\n{'─'*70}")
    print("🌱 INSPIRED BY TASK")
    print(f"{'─'*70}")
    print(f"  {scenario.seed_task_instruction[:150]}...")
    
    print(f"\n{'='*70}")
    print("🚀 SIMULATION STARTING - You are the AGENT helping this customer")
    print(f"{'='*70}\n")


def generate_scenario_for_env(
    env_name: str,
    model: str = "gpt-5.2",
    provider: str = "openai",
    task_ids: Optional[List[int]] = None,
) -> GeneratedScenario:
    """
    Convenience function to generate a scenario for an environment.
    
    Args:
        env_name: Name of the environment (retail, airline, etc.)
        model: LLM model to use
        provider: LLM provider
        task_ids: Optional list of task IDs to sample from (for focused training)
        
    Returns:
        GeneratedScenario with all data
    """
    generator = ScenarioGenerator(env_name, model, provider, task_ids=task_ids)
    return generator.generate()


def scenario_to_persona_data(scenario: GeneratedScenario) -> Dict[str, Any]:
    """
    Convert a GeneratedScenario to persona_data format for SimulatorCore.
    
    Args:
        scenario: The generated scenario
        
    Returns:
        Dict in persona_data format (includes augmented_data for DB injection)
    """
    return {
        "instruction": scenario.instruction,
        "user": scenario.user,
        scenario.data_key: scenario.data,
        "generated": True,
        "seed_instruction": scenario.seed_task_instruction,
        "timestamp": scenario.generation_timestamp,
        "env_name": scenario.env_name,
        "scenario_goal": scenario.scenario_goal,
        "augmented_data": scenario.augmented_data,  # Additional data to inject into DB
    }
