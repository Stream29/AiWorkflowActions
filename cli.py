import argparse
import os
from pathlib import Path
from typing import Optional, List

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from ai_workflow_action import (
    AiWorkflowAction, DifyWorkflowDslFile,
    DSLValidationReport, DifyWorkflowContextBuilder
)
from dsl_model import NodeType
from dsl_model.dsl import DifyWorkflowDSL

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

    def cmd_load(self, file_path: str):
        """Load a workflow file"""
        try:
            if not Path(file_path).exists():
                print(f"✗ File not found: {file_path}")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("✗ ANTHROPIC_API_KEY not found in environment variables")
                print("  This tool requires an Anthropic API key to function")
                return
            dsl_file = DifyWorkflowDslFile(file_path)
            self.ai_action = AiWorkflowAction(api_key=api_key, dsl_file=dsl_file)
            self.current_file = file_path
            info = self.ai_action.dsl_file.get_workflow_info()
            print(f"✓ Loaded and validated: {file_path}")
            print(f"  App: {info.app_name}")
            print(f"  Description: {info.description[:50]}{'...' if len(info.description) > 50 else ''}")
            print(f"  Nodes: {info.node_count}, Edges: {info.edge_count}")
            types_str = ', '.join([f"{k}({v})" for k, v in info.node_types.items()])
            print(f"  Types: {types_str}")
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
        except yaml.YAMLError as e:
            print(f"✗ Invalid YAML format: {e}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")

    def cmd_save(self, file_path: str):
        """Save the workflow"""
        if not self.ai_action:
            print("✗ No workflow loaded")
        try:
            self.ai_action.dsl_file.save(file_path)
            print(f"✓ Saved: {file_path}")
        except Exception as e:
            print(f"✗ Failed to save: {e}")

    def cmd_nodes(self):
        """List all nodes"""
        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        nodes = DifyWorkflowContextBuilder.extract_topological_sequence(self.ai_action.dsl_file)
        print(f"\n=== Nodes ({len(nodes)}) ===")

        for i, node in enumerate(nodes, 1):
            node_type = node.data.type
            node_title = self.ai_action.dsl_file.get_node(node.id).title
            conn_info = []
            for successor in node.successor_nodes:
                conn_info.append(f"→ {successor}")
            for predecessor in node.predecessor_nodes:
                conn_info.append(f"← {predecessor}")
            conn_str = f" [{', '.join(conn_info)}]" if conn_info else ""

            print(f"  {i}. [{node.id}] {node_title} ({node_type}){conn_str}")

    def cmd_generate(self, after_node_id: str, node_type: str):
        """Generate and add a new node using AI"""
        if not self.ai_action:
            print("✗ No workflow loaded")
        try:
            print(f"\n🤖 Generating {node_type} node after {after_node_id}...")
            node_id = self.ai_action.generate_node(after_node_id, NodeType(node_type))
            print(f"✓ Added node: {node_id}")
        except Exception as e:
            print(f"✗ Generation failed: {e}")

    def interactive_mode(self):
        """Run interactive command loop"""
        print("\n=== AiWorkflowActions CLI ===")
        print("Type 'help' for commands, 'exit' to quit\n")

        while True:
            try:
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

    def cmd_validate_resources(self):
        dsl_dir = project_root / 'resources' / 'Awesome-Dify-Workflow' / 'DSL'
        if not dsl_dir.exists():
            print(f"✗ DSL directory not found: {dsl_dir}")
            return
        yaml_files = list(dsl_dir.rglob('*.yml')) + list(dsl_dir.rglob('*.yaml'))
        if not yaml_files:
            print(f"✗ No YAML files found in: {dsl_dir}")
            return
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
            lines = [
                '# Dify DSL Validation Report',
                '',
                f"Scanned directory: `{dsl_dir}`",
                f"Total files: {len(yaml_files)} | Passed: {successes} | Failed: {len(failures)}",
                ''
            ]
            for item in failures:
                lines.append(f"## {item.file}")
                for err in item.errors:
                    lines.append(f"- {err}")
                lines.append('')
            content = '\n'.join(lines)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"⚠ Validation completed with failures. Report written to {report_path}")
            return
        else:
            print(f"✅ All {successes} files validated successfully with DifyWorkflowDSL.")
            return

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
    if args.validate_resources:
        cli.cmd_validate_resources()
        return
    if args.file:
        cli.cmd_load(args.file)
    # Enter interactive mode
    cli.interactive_mode()


if __name__ == '__main__':
    main()
