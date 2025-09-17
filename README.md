# AiWorkflowActions

AI-powered workflow node generation for Dify workflows.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set your Anthropic API key:
```bash
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Interactive Mode
```bash
uv run python cli.py
```

### Load File Directly
```bash
uv run python cli.py resources/SimpleDsl.yml
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `load <file>` | Load and validate workflow file |
| `nodes` | List all nodes in workflow |
| `detail --node <id>` | Show node details and JSON data |
| `generate --after <id> --type <type> [-m <message>]` | Generate new node with AI |
| `save <file>` | Save workflow to file |
| `help [command]` | Show help information |

## Generate Node Examples

```bash
# Basic generation
generate --after start_node --type llm

# With custom message
generate --after start_node --type code -m "Process JSON data"

# With title and message
generate --after start_node --type http-request --title "API Call" -m "Fetch user data"
```

## Supported Node Types

`start`, `end`, `answer`, `llm`, `code`, `http-request`, `tool`, `if-else`, `template-transform`, `variable-assigner`, `knowledge-retrieval`, `agent`, `iteration`, `parameter-extractor`, `question-classifier`, `iteration-start`, `loop-start`, `loop-end`, `variable-aggregator`, `document-extractor`, `list-operator`

## Validation

```bash
# Validate all DSL files
uv run python cli.py --validate-resources
```