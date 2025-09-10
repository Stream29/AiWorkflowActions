"""
Command-line interface for AiWorkflowActions
Pure UI layer that delegates to core modules
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from next_node_prediction import WorkflowCore, ContextBuilder, Validator, NodeGenerator


class CLI:
    """Interactive CLI for workflow management and AI generation"""

    def __init__(self):
        self.workflow = WorkflowCore()
        self.validator = Validator()
        self.context_builder = ContextBuilder()
        self.generator = None  # Lazy load if AI features are used
        self.current_file = None

    def cmd_load(self, file_path: str) -> bool:
        """Load a workflow file"""
        try:
            # Check file exists
            if not Path(file_path).exists():
                print(f"✗ File not found: {file_path}")
                return False

            self.workflow.load(file_path)
            self.current_file = file_path

            # Validate and show info
            info = self.workflow.get_workflow_info()
            print(f"✓ Loaded: {file_path}")
            print(f"  App: {info['app_name']}")
            print(f"  Description: {info['description'][:50]}{'...' if len(info['description']) > 50 else ''}")
            print(f"  Nodes: {info['node_count']}, Edges: {info['edge_count']}")

            # Show node types
            if info['node_types']:
                types_str = ', '.join([f"{k}({v})" for k, v in info['node_types'].items()])
                print(f"  Types: {types_str}")

            # Check if linear
            is_linear, error = self.validator.is_linear_workflow(self.workflow.workflow_data)
            if is_linear:
                print("  ✓ Linear workflow (AI generation supported)")
            else:
                print(f"  ⚠ Non-linear workflow: {error}")
                print("  Note: AI generation only works with linear workflows")

            return True
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
            return False
        except yaml.YAMLError as e:
            print(f"✗ Invalid YAML format: {e}")
            return False
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            return False

    def cmd_save(self, file_path: Optional[str] = None) -> bool:
        """Save the workflow"""
        try:
            output = self.workflow.save(file_path)
            print(f"✓ Saved: {output}")
            return True
        except Exception as e:
            print(f"✗ Failed to save: {e}")
            return False

    def cmd_validate(self) -> bool:
        """Validate the workflow"""
        if not self.workflow.workflow_data:
            print("✗ No workflow loaded")
            return False

        is_valid, results = self.validator.validate_workflow(self.workflow.workflow_data)

        print("\n=== Validation Results ===")

        if results['structure_errors']:
            print("Structure errors:")
            for error in results['structure_errors']:
                print(f"  - {error}")

        if results['node_errors']:
            print("Node errors:")
            for node_id, errors in results['node_errors'].items():
                print(f"  {node_id}:")
                for error in errors:
                    print(f"    - {error}")

        if results['graph_errors']:
            print("Graph errors:")
            for error in results['graph_errors']:
                print(f"  - {error}")

        if is_valid:
            print("✓ Workflow is valid!")
        else:
            print("✗ Workflow has errors")

        return is_valid

    def cmd_nodes(self) -> None:
        """List all nodes"""
        if not self.workflow.workflow_data:
            print("✗ No workflow loaded")
            return

        nodes = self.workflow.nodes
        print(f"\n=== Nodes ({len(nodes)}) ===")

        for i, node in enumerate(nodes, 1):
            node_id = node.get('id')
            node_data = node.get('data', {})
            node_type = node_data.get('type')
            node_title = node_data.get('title', 'Untitled')

            connections = self.workflow.get_node_connections(node_id)
            conn_info = []
            if connections['incoming']:
                conn_info.append(f"← {len(connections['incoming'])}")
            if connections['outgoing']:
                conn_info.append(f"→ {len(connections['outgoing'])}")
            conn_str = f" [{', '.join(conn_info)}]" if conn_info else ""

            print(f"  {i}. [{node_id}] {node_title} ({node_type}){conn_str}")

    def cmd_generate(self, after_node_id: str, node_type: str) -> Optional[str]:
        """Generate and add a new node using AI"""
        if not self.workflow.workflow_data:
            print("✗ No workflow loaded")
            return None

        # Check if workflow is linear
        is_linear, error = self.validator.is_linear_workflow(self.workflow.workflow_data)
        if not is_linear:
            print(f"✗ AI generation requires linear workflow: {error}")
            return None

        # Initialize generator if needed
        if not self.generator:
            try:
                self.generator = NodeGenerator()
            except Exception as e:
                print(f"✗ Failed to initialize AI generator: {e}")
                return None

        try:
            print(f"\n🤖 Generating {node_type} node after {after_node_id}...")

            # Build context
            context = self.context_builder.build_context(
                self.workflow.workflow_data,
                target_position=after_node_id
            )

            # Get schema for the node type
            schema = self.validator.get_node_schema(node_type)

            # Generate with retry on validation failure
            max_attempts = 3
            for attempt in range(max_attempts):
                print(f"  Attempt {attempt + 1}/{max_attempts}...")

                # Generate node data
                previous_errors = None
                node_data = self.generator.generate_node(
                    node_type, context, schema, previous_errors
                )

                # Validate generated data
                is_valid, validation_errors = self.validator.validate_node_data(
                    node_type, node_data
                )

                if is_valid:
                    print("  ✓ Generated valid node data")

                    # Create full node structure
                    new_node = {
                        'data': {
                            'type': node_type,
                            **node_data
                        },
                        'type': 'custom',
                        'selected': False,
                        'sourcePosition': 'right',
                        'targetPosition': 'left'
                    }

                    # Add to workflow
                    node_id = self.workflow.add_node_after(after_node_id, new_node)
                    print(f"✓ Added node: {node_id}")
                    return node_id
                else:
                    print(f"  ✗ Validation failed:")
                    for error in validation_errors[:3]:  # Show first 3 errors
                        print(f"    - {error}")

            print(f"✗ Failed to generate valid node after {max_attempts} attempts")
            return None

        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return None

    def cmd_auto_next(self, node_type: Optional[str] = None) -> Optional[str]:
        """AI auto-generate and add the next most suitable node"""
        if not self.workflow.workflow_data:
            print("✗ No workflow loaded")
            return None

        # Check if workflow is linear
        is_linear, error = self.validator.is_linear_workflow(self.workflow.workflow_data)
        if not is_linear:
            print(f"✗ AI generation requires linear workflow: {error}")
            return None

        # Find terminal nodes
        terminal_nodes = self.workflow.get_terminal_nodes()
        if not terminal_nodes:
            print("✗ No terminal nodes found in workflow")
            return None

        if len(terminal_nodes) > 1:
            print(f"⚠ Multiple terminal nodes found, using first: {terminal_nodes[0]['id']}")

        terminal_node = terminal_nodes[0]
        terminal_id = terminal_node['id']
        terminal_type = terminal_node.get('data', {}).get('type')

        print(f"\n🤖 Analyzing workflow ending at: {terminal_id} ({terminal_type})")

        # Get recommended node types if none specified
        if not node_type:
            recommended_types = self.get_recommended_node_types(terminal_type)
            if not recommended_types:
                print("✗ No suitable node type recommendations")
                return None

            print(f"Recommended node types: {', '.join(recommended_types)}")
            node_type = recommended_types[0]  # Use first recommendation
            print(f"Using: {node_type}")

        # Generate and add the node
        return self.cmd_generate(terminal_id, node_type)

    def get_recommended_node_types(self, last_node_type: str) -> List[str]:
        """Get recommended next node types based on the last node"""
        recommendations = {
            'start': ['llm', 'code', 'http-request', 'variable-assigner'],
            'llm': ['code', 'end', 'if-else', 'variable-assigner', 'parameter-extractor'],
            'code': ['end', 'llm', 'if-else', 'variable-assigner'],
            'http-request': ['code', 'llm', 'parameter-extractor', 'end'],
            'variable-assigner': ['llm', 'code', 'end', 'if-else'],
            'parameter-extractor': ['llm', 'code', 'end'],
            'if-else': ['llm', 'code', 'end'],
        }
        return recommendations.get(last_node_type, ['end'])

    def cmd_remove(self, node_id: str) -> bool:
        """Remove a node"""
        if not self.workflow.workflow_data:
            print("✗ No workflow loaded")
            return False

        try:
            if self.workflow.remove_node(node_id):
                print(f"✓ Removed node: {node_id}")
                return True
            else:
                print(f"✗ Node not found: {node_id}")
                return False
        except Exception as e:
            print(f"✗ Failed to remove node: {e}")
            return False

    def interactive_mode(self):
        """Run interactive command loop"""
        print("\n=== AiWorkflowActions CLI ===")
        print("Type 'help' for commands, 'exit' to quit\n")

        while True:
            try:
                # Show prompt
                prompt = f"[{Path(self.current_file).name if self.current_file else 'no file'}]> "
                command = input(prompt).strip()

                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()

                if cmd in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break

                elif cmd == 'help':
                    self.show_help()

                elif cmd == 'load' and len(parts) > 1:
                    self.cmd_load(parts[1])

                elif cmd == 'save':
                    path = parts[1] if len(parts) > 1 else None
                    self.cmd_save(path)

                elif cmd == 'validate':
                    self.cmd_validate()

                elif cmd == 'nodes':
                    self.cmd_nodes()

                elif cmd == 'generate' and len(parts) >= 3:
                    self.cmd_generate(parts[1], parts[2])

                elif cmd == 'remove' and len(parts) > 1:
                    self.cmd_remove(parts[1])

                elif cmd in ['auto-next', 'auto_next', 'auto']:
                    node_type = parts[1] if len(parts) > 1 else None
                    self.cmd_auto_next(node_type)

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    def show_help(self):
        """Show help message"""
        print("""
Commands:
  load <file>              - Load workflow file
  save [file]              - Save workflow (optional: new file)
  validate                 - Validate workflow
  nodes                    - List all nodes
  generate <after> <type>  - Generate and add node using AI
  auto-next [type]         - 🚀 AI auto-generate next suitable node
  remove <node_id>         - Remove a node
  help                     - Show this help
  exit                     - Quit

MVP Examples (Real workflow editing):
  load resources/SimpleDsl.yml
  nodes                    # View current workflow
  auto-next                # AI generates best next node automatically
  auto-next code           # Force generate specific node type
  validate                 # Check workflow integrity
  save enhanced.yml        # Save with new node

Traditional Examples:
  generate start llm       # Generate after specific node
  generate llm-1 code      # Generate code node after llm-1
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AiWorkflowActions - AI-powered workflow node generation'
    )

    parser.add_argument('file', nargs='?', help='Workflow file to load')
    parser.add_argument('--generate', nargs=2, metavar=('AFTER', 'TYPE'),
                       help='Generate node after specified node')
    parser.add_argument('--validate', action='store_true',
                       help='Validate workflow and exit')
    parser.add_argument('--output', '-o', help='Output file for save')

    args = parser.parse_args()

    cli = CLI()

    # Load file if provided
    if args.file:
        if not cli.cmd_load(args.file):
            return 1

    # Execute single command if provided
    if args.validate:
        return 0 if cli.cmd_validate() else 1

    if args.generate:
        after_node, node_type = args.generate
        if not cli.cmd_generate(after_node, node_type):
            return 1
        if args.output:
            cli.cmd_save(args.output)
        return 0

    # Enter interactive mode
    cli.interactive_mode()
    return 0


if __name__ == '__main__':
    exit(main())