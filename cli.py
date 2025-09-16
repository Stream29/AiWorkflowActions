"""
Command-line interface for AiWorkflowActions
Pure UI layer that delegates to core modules
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from pydantic import ValidationError
from dsl_model.dsl import DifyWorkflowDSL
from ai_workflow_action import (
    AiWorkflowAction, DifyWorkflowDslFile,
    WorkflowInfo, WorkflowValidationResult, LinearityCheck, 
    NodeInfo, NodeConnection, DSLValidationSummary, DSLValidationReport
)

# Load environment variables from project root
project_root = Path(__file__).resolve().parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)


class CLI:
    """Interactive CLI for workflow management and AI generation"""

    def __init__(self):
        # Use new layered architecture
        self.ai_action: Optional[AiWorkflowAction] = None
        self.current_file = None

    def cmd_load(self, file_path: str) -> bool:
        """Load a workflow file"""
        try:
            # Check file exists
            if not Path(file_path).exists():
                print(f"✗ File not found: {file_path}")
                return False

            # Get API key from environment - required for operation
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("✗ ANTHROPIC_API_KEY not found in environment variables")
                print("  This tool requires an Anthropic API key to function")
                return False

            # Create DSL file and validate immediately
            dsl_file = DifyWorkflowDslFile(file_path)
            
            # Validate DSL before proceeding
            validation_result = dsl_file.validate_workflow()
            if not validation_result.is_valid:
                print(f"✗ Invalid workflow file: {file_path}")
                if validation_result.structure_errors:
                    for error in validation_result.structure_errors:
                        print(f"  - {error}")
                if validation_result.node_errors:
                    for node_id, errors in validation_result.node_errors.items():
                        print(f"  {node_id}: {'; '.join(errors)}")
                if validation_result.graph_errors:
                    for error in validation_result.graph_errors:
                        print(f"  - {error}")
                return False
            
            # Only create AI action if DSL is valid
            self.ai_action = AiWorkflowAction(api_key, dsl_file)
            self.current_file = file_path

            # Show workflow info
            info = self.ai_action.dsl_file.get_workflow_info()
            print(f"✓ Loaded and validated: {file_path}")
            print(f"  App: {info.app_name}")
            print(f"  Description: {info.description[:50]}{'...' if len(info.description) > 50 else ''}")
            print(f"  Nodes: {info.node_count}, Edges: {info.edge_count}")

            # Show node types
            if info.node_types:
                types_str = ', '.join([f"{k}({v})" for k, v in info.node_types.items()])
                print(f"  Types: {types_str}")

            # Check if linear (AI compatibility)
            linearity_check = self.ai_action.dsl_file.is_linear_workflow()
            if linearity_check.is_linear:
                print("  ✓ Linear workflow (AI generation supported)")
            else:
                print(f"  ⚠ Non-linear workflow: {linearity_check.error_message}")
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
        if not self.ai_action:
            print("✗ No workflow loaded")
            return False
            
        try:
            output = self.ai_action.save_workflow(file_path)
            print(f"✓ Saved: {output}")
            return True
        except Exception as e:
            print(f"✗ Failed to save: {e}")
            return False


    def cmd_nodes(self) -> None:
        """List all nodes"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        nodes = self.ai_action.dsl_file.nodes
        print(f"\n=== Nodes ({len(nodes)}) ===")

        for i, node in enumerate(nodes, 1):
            node_id = node.id
            node_data = node.data
            node_type = node_data.type
            node_title = getattr(node_data, 'title', 'Untitled')

            connections = self.ai_action.dsl_file.get_node_connections(node_id)
            conn_info = []
            if connections.incoming:
                conn_info.append(f"← {len(connections.incoming)}")
            if connections.outgoing:
                conn_info.append(f"→ {len(connections.outgoing)}")
            conn_str = f" [{', '.join(conn_info)}]" if conn_info else ""

            print(f"  {i}. [{node_id}] {node_title} ({node_type}){conn_str}")

    def cmd_generate(self, after_node_id: str, node_type: str) -> Optional[str]:
        """Generate and add a new node using AI"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return None

        try:
            print(f"\n🤖 Generating {node_type} node after {after_node_id}...")
            
            # Use the AI action to generate node (simplified interface)
            node_id = self.ai_action.generate_node(after_node_id, node_type)
            
            if node_id:
                print(f"✓ Added node: {node_id}")
                return node_id
            else:
                print("✗ Failed to generate valid node after multiple attempts")
                return None

        except Exception as e:
            print(f"✗ Generation failed: {e}")
            return None



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
                    if len(parts) > 1:
                        self.cmd_save(parts[1])
                    else:
                        print("Usage: save <file_path>")

                elif cmd == 'nodes':
                    self.cmd_nodes()

                elif cmd == 'generate' and len(parts) >= 4 and parts[1] == 'after':
                    self.cmd_generate(parts[2], parts[3])

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    def cmd_validate_resources(self) -> bool:
        """Validate all DSL files under resources/Awesome-Dify-Workflow/DSL using DifyWorkflowDSL.
        If any errors are found, write a markdown report to project root.
        """
        project_root = Path(__file__).resolve().parent
        dsl_dir = project_root / 'resources' / 'Awesome-Dify-Workflow' / 'DSL'
        if not dsl_dir.exists():
            print(f"✗ DSL directory not found: {dsl_dir}")
            return False

        yaml_files = list(dsl_dir.rglob('*.yml')) + list(dsl_dir.rglob('*.yaml'))
        if not yaml_files:
            print(f"✗ No YAML files found in: {dsl_dir}")
            return False

        print(f"Scanning {len(yaml_files)} DSL files under {dsl_dir} ...")
        failures: List[DSLValidationReport] = []
        successes = 0

        for yf in yaml_files:
            rel = yf.relative_to(project_root)
            try:
                with open(yf, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data is None:
                    raise ValueError('Empty YAML file')
                # Parse with Pydantic DSL
                DifyWorkflowDSL.model_validate(data)
                successes += 1
                print(f"✓ {rel}")
            except ValidationError as e:
                err_list = [f"{' -> '.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
                failures.append(DSLValidationReport(
                    file=str(rel),
                    errors=err_list,
                ))
                print(f"✗ {rel} ({len(err_list)} errors)")
            except Exception as e:
                failures.append(DSLValidationReport(
                    file=str(rel),
                    errors=[str(e)],
                ))
                print(f"✗ {rel} (exception)")

        if failures:
            report_path = project_root / 'DIFY_DSL_VALIDATION_REPORT.md'
            lines = []
            lines.append('# Dify DSL Validation Report')
            lines.append('')
            lines.append(f"Scanned directory: `{dsl_dir}`")
            lines.append(f"Total files: {len(yaml_files)} | Passed: {successes} | Failed: {len(failures)}")
            lines.append('')
            for item in failures:
                lines.append(f"## {item.file}")
                for err in item.errors:
                    lines.append(f"- {err}")
                lines.append('')
            content = '\n'.join(lines)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"⚠ Validation completed with failures. Report written to {report_path}")
            return False
        else:
            print(f"✅ All {successes} files validated successfully with DifyWorkflowDSL.")
            return True

    def show_help(self):
        """Show help message"""
        print("""
Commands:
  load <file_path>         - Load and validate workflow file
  save <file_path>         - Save workflow to file
  nodes                    - List all nodes in workflow
  generate after <node_id> <node_type> - Generate and add node using AI
  help                     - Show this help
  exit                     - Quit

Examples:
  load resources/SimpleDsl.yml
  nodes
  generate after start llm
  generate after llm-1 code
  save enhanced.yml
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AiWorkflowActions - AI-powered workflow node generation'
    )

    parser.add_argument('file', nargs='?', help='Workflow file to load')
    parser.add_argument('--validate-resources', action='store_true',
                       help='Validate all DSL files in resources and write report if any errors')

    args = parser.parse_args()

    cli = CLI()

    # Execute single command if provided
    if args.validate_resources:
        return 0 if cli.cmd_validate_resources() else 1

    # Load file if provided, then enter interactive mode
    if args.file:
        if not cli.cmd_load(args.file):
            return 1

    # Enter interactive mode
    cli.interactive_mode()
    return 0


if __name__ == '__main__':
    exit(main())