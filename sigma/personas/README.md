# Personas Directory

This directory stores generated persona files created by `sigma.persona_creator`.

## Creating a Persona

```bash
python -m sigma.persona_creator --env retail --model gpt-4o --provider openai
```

## Using a Persona

```bash
python -m sigma --env retail --user-model gpt-4o --user-provider openai \
    --persona-file sigma/personas/<persona_name>_retail.json
```

## Persona File Format

```json
{
    "name": "persona_name",
    "env": "retail",
    "created_at": "2024-01-01T12:00:00",
    "instruction": "You are John Doe...",
    "user": { ... user profile ... },
    "orders": { ... orders data ... }
}
```

For airline environment, `orders` is replaced with `reservations`.
