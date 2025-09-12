"""
Command-line interface for AiWorkflowActions
Pure UI layer that delegates to core modules
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import ValidationError
from dsl_model.dsl import DifyWorkflowDSL
from ai_workflow_action import (
    AiWorkflowAction, DifyWorkflowDslFile,
    WorkflowInfo, WorkflowValidationResult, LinearityCheck, 
    NodeInfo, NodeConnection, DSLValidationSummary, DSLValidationReport
)


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

            # Create DSL file and AI action with new architecture
            dsl_file = DifyWorkflowDslFile(file_path)
            self.ai_action = AiWorkflowAction(dsl_file)
            self.current_file = file_path

            # Show workflow info
            info = dsl_file.get_workflow_info()
            print(f"✓ Loaded: {file_path}")
            print(f"  App: {info.app_name}")
            print(f"  Description: {info.description[:50]}{'...' if len(info.description) > 50 else ''}")
            print(f"  Nodes: {info.node_count}, Edges: {info.edge_count}")

            # Show node types
            if info.node_types:
                types_str = ', '.join([f"{k}({v})" for k, v in info.node_types.items()])
                print(f"  Types: {types_str}")

            # Check if linear (AI compatibility)
            linearity_check = dsl_file.is_linear_workflow()
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

    def cmd_validate(self) -> bool:
        """Validate the workflow"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return False

        validation_result = self.ai_action.dsl_file.validate_workflow()

        print("\n=== Validation Results ===")

        if validation_result.structure_errors:
            print("Structure errors:")
            for error in validation_result.structure_errors:
                print(f"  - {error}")

        if validation_result.node_errors:
            print("Node errors:")
            for node_id, errors in validation_result.node_errors.items():
                print(f"  {node_id}:")
                for error in errors:
                    print(f"    - {error}")

        if validation_result.graph_errors:
            print("Graph errors:")
            for error in validation_result.graph_errors:
                print(f"  - {error}")

        if validation_result.is_valid:
            print("✓ Workflow is valid!")
        else:
            print("✗ Workflow has errors")

        return validation_result.is_valid

    def cmd_nodes(self) -> None:
        """List all nodes"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        nodes = self.ai_action.dsl_file.nodes
        print(f"\n=== Nodes ({len(nodes)}) ===")

        for i, node in enumerate(nodes, 1):
            node_id = node.get('id')
            node_data = node.get('data', {})
            node_type = node_data.get('type')
            node_title = node_data.get('title', 'Untitled')

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

    def cmd_auto_next(self, node_type: Optional[str] = None) -> Optional[str]:
        """AI auto-generate and add the next most suitable node"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return None

        try:
            # Get analysis to show user what's happening
            if not node_type:
                analysis = self.ai_action.analyze_workflow()
                completion_analysis = analysis.get("completion_analysis", {})
                
                if completion_analysis.get("is_complete", False):
                    print("✓ Workflow appears to be complete")
                    return None
                
                last_node = completion_analysis.get("last_node", {})
                recommendations = completion_analysis.get("recommendations", [])
                
                print(f"\n🤖 Analyzing workflow ending at: {last_node.get('id')} ({last_node.get('type')})")
                
                if recommendations:
                    print(f"Recommended node types: {', '.join(recommendations)}")
                    node_type = recommendations[0]
                    print(f"Using: {node_type}")
                else:
                    print("✗ No suitable node type recommendations")
                    return None

            # Use the AI action's auto-generation
            node_id = self.ai_action.auto_generate_next_node(node_type)
            
            if node_id:
                print(f"✓ Added node: {node_id}")
                return node_id
            else:
                print("✗ Failed to generate next node")
                return None

        except Exception as e:
            print(f"✗ Auto-generation failed: {e}")
            return None

    def cmd_remove(self, node_id: str) -> bool:
        """Remove a node"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return False

        try:
            if self.ai_action.dsl_file.remove_node(node_id):
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
  load <file>              - Load workflow file
  save [file]              - Save workflow (optional: new file)
  validate                 - Validate workflow
  validate-resources       - Validate all DSL files in resources and write report if any errors
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
  validate-resources       # Validate all sample DSLs and generate report if needed
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
    parser.add_argument('--validate-resources', action='store_true',
                       help='Validate all DSL files in resources and write report if any errors')
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

    if args.validate_resources:
        return 0 if cli.cmd_validate_resources() else 1

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