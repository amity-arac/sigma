# Copyright Amity

import abc
import enum
import os
import time
from litellm import completion
from dotenv import load_dotenv

from typing import Optional, List, Dict, Any, Union

# Load .env file before reading environment variables
load_dotenv()

# Debug flag - set to True to see prompts sent to LLM
DEBUG_USER_PROMPTS = os.environ.get("DEBUG_USER_PROMPTS", "false").lower() == "true"

# Path to shared prompts directory
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def load_base_prompt(filename: str) -> str:
    """Load a base prompt from the shared prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()
    raise FileNotFoundError(f"Base prompt not found: {prompt_path}")


def load_env_guidelines(env_path: Optional[str], filename: str) -> str:
    """Load environment-specific guidelines, or return empty string."""
    if env_path is None:
        return ""
    
    guidelines_path = os.path.join(env_path, filename)
    if os.path.exists(guidelines_path):
        with open(guidelines_path, "r") as f:
            return f.read()
    return ""


def build_user_prompt(env_path: Optional[str]) -> str:
    """Build complete user prompt by combining base + env-specific guidelines."""
    base_prompt = load_base_prompt("user_base.md")
    env_guidelines = load_env_guidelines(env_path, "user_guidelines.md")
    return base_prompt.replace("{env_guidelines}", env_guidelines)


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}
    messages: List[Dict[str, Any]]

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset the conversation state. Does NOT generate a message."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Generate the next message from the simulated user."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self._instruction: Optional[str] = None

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = instruction
        self.messages = []

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        prompt = self._instruction if self._instruction else "Enter your message"
        return input(f"{prompt}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return 0


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str, env_path: Optional[str] = None) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.env_path = env_path
        self._prompt_template: Optional[str] = None
        # Don't call reset() here - it will be called explicitly with instruction later
        # This avoids a wasted LLM call during initialization

    def _load_prompt_template(self) -> str:
        """Load the prompt template by combining base + env-specific guidelines."""
        if self._prompt_template is None:
            self._prompt_template = build_user_prompt(self.env_path)
        return self._prompt_template

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        if DEBUG_USER_PROMPTS:
            print(f"\n{'='*80}")
            print(f"[DEBUG USER PROMPT] Sending to {self.model} via {self.provider}:")
            print(f"{'='*80}")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"\n--- Message {i+1} [{role.upper()}] ---")
                print(content)  # Full content, no truncation
            print(f"\n{'='*80}\n")
        
        start_time = time.time()
        res = completion(
            model=self.model, custom_llm_provider=self.provider, messages=messages
        )
        elapsed = time.time() - start_time
        
        message = res.choices[0].message
        self.messages.append(message.model_dump())
        self.total_cost = res._hidden_params.get("response_cost", 0)
        
        if DEBUG_USER_PROMPTS:
            print(f"\n{'='*80}")
            print(f"[DEBUG USER RESPONSE] ({elapsed:.2f}s):")
            print(f"{'='*80}")
            print(message.content)  # Full response, no truncation
            print(f"{'='*80}\n")
        
        return message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = instruction if instruction else "(No specific instruction provided)"
        template = self._load_prompt_template()
        return template.replace("{instruction}", instruction_display)

    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset the conversation state with a new instruction.
        
        Note: This does NOT generate a message. Call generate_next_message() separately
        to get the initial user message after reset.
        """
        # Build system prompt with instruction
        system_prompt = self.build_system_prompt(instruction=instruction)
        
        # Start fresh - user initiates the conversation
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "You are now starting a customer service chat. Send your first message to the agent."},
        ]

    def step(self, content: str) -> str:
        # In the user simulation, the LLM plays the customer role:
        # - "user" role = input/prompts to the model (what the agent says)
        # - "assistant" role = model's output (what the customer says)
        # So the agent's message should be "user" role, not "assistant"
        self.messages.append({"role": "user", "content": f"The customer service agent says: {content}\n\nGenerate your response as the customer."})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


# Alias for backward compatibility
ReactUserSimulationEnv = LLMUserSimulationEnv


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 3, env_path: Optional[str] = None) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.total_cost = 0.0
        self.messages: List[Dict[str, Any]] = []
        self.env_path = env_path
        self._prompt_template: Optional[str] = None
        # Don't call reset() here - it will be called explicitly with instruction later

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        attempts = 0
        cur_message = None
        while attempts < self.max_attempts:
            if DEBUG_USER_PROMPTS:
                print(f"\n{'='*80}")
                print(f"[DEBUG USER PROMPT] VerifyUserSimulationEnv.generate_next_message (attempt {attempts+1}/{self.max_attempts}) - Sending to {self.model} via {self.provider}:")
                print(f"{'='*80}")
                for i, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    print(f"\n--- Message {i+1} [{role.upper()}] ---")
                    print(content)  # Full content, no truncation
                print(f"{'='*80}\n")
            
            res = completion(
                model=self.model, custom_llm_provider=self.provider, messages=messages
            )
            cur_message = res.choices[0].message
            self.total_cost = res._hidden_params["response_cost"]
            
            if DEBUG_USER_PROMPTS:
                print(f"\n{'='*80}")
                print(f"[DEBUG USER PROMPT] VerifyUserSimulationEnv.generate_next_message - Response:")
                print(f"{'='*80}")
                print(cur_message.content)  # Full response, no truncation
                print(f"{'='*80}\n")
            
            if verify(self.model, self.provider, cur_message, messages):
                self.messages.append(cur_message.model_dump())
                return cur_message.content
            attempts += 1
        assert cur_message is not None
        return cur_message.content

    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset the conversation state. Does NOT generate a message."""
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    
    if DEBUG_USER_PROMPTS:
        print(f"\n{'='*80}")
        print(f"[DEBUG USER PROMPT] verify() - Sending to {model} via {provider}:")
        print(f"{'='*80}")
        print(f"\n--- Message 1 [USER] ---")
        print(prompt)  # Full content, no truncation
        print(f"{'='*80}\n")
    
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    result_content = res.choices[0].message.content
    result = "true" in result_content.lower()
    
    if DEBUG_USER_PROMPTS:
        print(f"\n{'='*80}")
        print(f"[DEBUG USER PROMPT] verify() - Response:")
        print(f"{'='*80}")
        print(result_content)  # Full response, no truncation
        print(f"Result: {result}")
        print(f"{'='*80}\n")
    
    return result


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    
    if DEBUG_USER_PROMPTS:
        print(f"\n{'='*80}")
        print(f"[DEBUG USER PROMPT] reflect() - Sending to {model} via {provider}:")
        print(f"{'='*80}")
        print(f"\n--- Message 1 [USER] ---")
        print(prompt)  # Full content, no truncation
        print(f"{'='*80}\n")
    
    res = completion(
        model=model,
        custom_llm_provider=provider,
        messages=[{"role": "user", "content": prompt}],
    )
    result_content = res.choices[0].message.content
    _, response_part = result_content.split("Response:")
    
    if DEBUG_USER_PROMPTS:
        print(f"\n{'='*80}")
        print(f"[DEBUG USER PROMPT] reflect() - Response:")
        print(f"{'='*80}")
        print(result_content)  # Full response, no truncation
        print(f"{'='*80}\n")
    
    return response_part.strip()


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 2, env_path: Optional[str] = None) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.total_cost = 0.0
        self.messages: List[Dict[str, Any]] = []
        self.env_path = env_path
        self._prompt_template: Optional[str] = None
        # Don't call reset() here - it will be called explicitly with instruction later

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cur_messages = messages.copy()
        initial_response = super().generate_next_message(cur_messages)
        if verify(self.model, self.provider, initial_response, cur_messages):
            return initial_response
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(
                self.model, self.provider, initial_response, cur_messages
            )
            cur_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(cur_messages)
            if verify(self.model, self.provider, new_response, cur_messages):
                return new_response
            attempts += 1
        return initial_response

    def reset(self, instruction: Optional[str] = None) -> None:
        """Reset the conversation state. Does NOT generate a message."""
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]

    # step() and get_total_cost() are inherited from LLMUserSimulationEnv


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
    env_path: Optional[str] = None,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider, env_path=env_path)
    elif user_strategy == UserStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider, env_path=env_path)
    elif user_strategy == UserStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider, env_path=env_path)
    elif user_strategy == UserStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider, env_path=env_path)
    raise ValueError(f"Unknown user strategy {user_strategy}")
