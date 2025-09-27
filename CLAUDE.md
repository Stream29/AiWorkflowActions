## Type Checking

```bash
ty check --exclude resources
```

Ensure type checking passes after every edit by running the above command.

## Dependency Management

```bash
uv run <script>
```

```bash
uv sync
```

```bash
uv add <package>
```

Use `uv` whenever possible, avoid using `pip` or `python` directly.

## Type Safety

Use strict type annotations.

Avoid using `Any` and `Dict` whenever possible.

Never use `hasattr` and `getattr` anywhere.

Use Pydantic models whenever possible.

## Reference Materials

The project's `/resources` directory contains:

- `SimpleDsl.yml` - Simple DSL file
- `Awesome-Dify-Workflow` - Project containing numerous DSL files
- `dify` - Official Dify repository containing official parsing and editing logic, data structures

## Architecture Design

The project uses a layered architecture design with four layers from bottom to top:

### Layer 1: DifyWorkflowDSL (dsl_model)
- Bottom-most data structure layer
- Defines complete Pydantic models for Dify workflows
- Location: `src/dsl_model/`
- Responsibility: Encapsulates low-level data structures and validation logic

### Layer 2: DifyWorkflowDslFile
- Workflow file operations layer
- Wraps DifyWorkflowDSL, provides basic operations (load, save, modify, validate)
- Implements RAII pattern for resource management
- Location: `src/ai_workflow_action/dsl_file.py`
- Responsibility: File I/O, basic workflow operations, data validation

### Layer 3: AiWorkflowAction
- AI operations layer
- Provides AI-supported workflow operations through DifyWorkflowContextBuilder
- Manages Anthropic API resources
- Location: `src/ai_workflow_action/ai_workflow_action.py`
- Responsibility: AI node generation, smart recommendations, workflow analysis

### Layer 4: CLI
- User interface layer
- Provides command-line interface using AiWorkflowAction
- Based on `cmd.Cmd` + `argparse` architecture, supports complex command parameter parsing
- Location: `cli.py`
- Responsibility: User interaction, command parsing, result display
- Features:
  - Auto-completion and command history
  - Built-in help system
  - Elegant error handling
  - Support for complex parameters (options, flags, positional arguments, etc.)
  - Auto-generated usage instructions

### Supporting Component: DifyWorkflowContextBuilder
- Specialized context builder
- Provides high-quality prompts and context for AI operations
- Location: `src/ai_workflow_action/context_builder.py`
- Responsibility: Workflow analysis, context extraction, AI prompt generation

## CLI Commands

### Basic Commands
```bash
# Start interactive mode
uv run python cli.py

# Load file directly
uv run python cli.py resources/SimpleDsl.yml

# Validate all resource files
uv run python cli.py --validate-resources
```

### Interactive Commands

```
load <file_path>                          - Load and validate workflow file
save <file_path>                          - Save workflow to file
nodes [--verbose]                         - List all nodes in workflow
  --verbose, -v                           - Show detailed connection information
detail --node <node_id>                  - Show detailed information and JSON data for specified node
  --node <node_id>                        - Node ID to view details for
generate --after <node_id> --type <type> [--title <title>] [-m <message>]  - Generate new node using AI
  --after <node_id>                       - Add new node after specified node
  --type <node_type>                      - Node type (e.g., llm, code, http-request)
  --title <title>                         - Optional custom node title
  -m, --message <message>                 - Custom generation intent/instruction
remove --single <node_id>                - Remove a single node and all edges connected to it
  --single <node_id>                     - ID of the single node to remove
remove --after <node_id>                 - Remove all nodes after the specified node
  --after <node_id>                      - ID of the node whose successors to remove
validate_resources [--dir <directory>]   - Validate DSL files
  --dir <directory>                       - Custom validation directory
help [command]                            - Show help information
quit/exit                                 - Exit program
```

## Supported Node Types

The following node types are supported:

- `start` - Start node
- `end` - End node
- `answer` - Answer node
- `llm` - LLM node
- `code` - Code execution node
- `http-request` - HTTP request node
- `tool` - Tool call node
- `if-else` - Conditional branch node
- `template-transform` - Template transformation node
- `variable-assigner` - Variable assignment node
- `knowledge-retrieval` - Knowledge retrieval node
- `agent` - Agent node
- `iteration` - Iteration node
- `parameter-extractor` - Parameter extraction node
- `question-classifier` - Question classification node
- `iteration-start` - Iteration start node
- `loop-start` - Loop start node
- `loop-end` - Loop end node
- `variable-aggregator` - Variable aggregator node
- `document-extractor` - Document extraction node
- `list-operator` - List operator node
- `` - Note node, marked as empty

## Environment Variables

The project requires the following environment variables to be configured:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

This variable should be placed in the `.env` file in the project root directory.
