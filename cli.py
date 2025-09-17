import argparse
import cmd
import json
import os
import shlex
import sys
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


class CLI(cmd.Cmd):
    """Interactive CLI for workflow management and AI generation"""

    intro = "\n=== AiWorkflowActions CLI ===\nType 'help' or '?' for commands, 'quit' to exit\n"

    def __init__(self):
        super().__init__()
        # Use new layered architecture
        self.ai_action: Optional[AiWorkflowAction] = None
        self.current_file = None
        self._update_prompt()

    def _update_prompt(self):
        """Update the prompt to show current file"""
        file_name = Path(self.current_file).name if self.current_file else 'no file'
        self.prompt = f"[{file_name}]> "

    def _parse_args(self, parser: argparse.ArgumentParser, args: str) -> Optional[argparse.Namespace]:
        """Parse arguments using argparse and handle errors gracefully"""
        try:
            # Use shlex to properly split the command line
            arg_list = shlex.split(args) if args else []
            return parser.parse_args(arg_list)
        except SystemExit:
            # argparse calls sys.exit on error, catch and return None
            return None
        except Exception as e:
            print(f"Error parsing arguments: {e}")
            return None

    def do_load(self, args: str):
        """Load a workflow file

        Usage: load <file_path>

        Arguments:
            file_path: Path to the workflow YAML file to load
        """
        parser = argparse.ArgumentParser(description="Load a workflow file", prog='load')
        parser.add_argument('file_path', help='Path to the workflow YAML file')

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        file_path = parsed_args.file_path
        try:
            if not Path(file_path).exists():
                print(f"✗ File not found: {file_path}")
                return
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("✗ ANTHROPIC_API_KEY not found in environment variables")
                print("  This tool requires an Anthropic API key to function")
                return
            dsl_file = DifyWorkflowDslFile(file_path)
            self.ai_action = AiWorkflowAction(api_key=api_key, dsl_file=dsl_file)
            self.current_file = file_path
            self._update_prompt()
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

    def do_save(self, args: str):
        """Save the workflow

        Usage: save <file_path>

        Arguments:
            file_path: Path where to save the workflow YAML file
        """
        parser = argparse.ArgumentParser(description="Save the workflow", prog='save')
        parser.add_argument('file_path', help='Path where to save the workflow YAML file')

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        try:
            self.ai_action.dsl_file.save(parsed_args.file_path)
            print(f"✓ Saved: {parsed_args.file_path}")
        except Exception as e:
            print(f"✗ Failed to save: {e}")

    def do_nodes(self, args: str):
        """List all nodes

        Usage: nodes [--verbose]

        Options:
            --verbose, -v: Show detailed connection information
        """
        parser = argparse.ArgumentParser(description="List all nodes in the workflow", prog='nodes')
        parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed connection information')

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        nodes = DifyWorkflowContextBuilder.extract_topological_sequence(self.ai_action.dsl_file)
        print(f"\n=== Nodes ({len(nodes)}) ===")

        for i, node in enumerate(nodes, 1):
            node_type = node.data.type
            node_title = self.ai_action.dsl_file.get_node(node.id).data.title

            if parsed_args.verbose:
                conn_info = []
                for successor in node.successor_nodes:
                    conn_info.append(f"→ {successor}")
                for predecessor in node.predecessor_nodes:
                    conn_info.append(f"← {predecessor}")
                conn_str = f" [{', '.join(conn_info)}]" if conn_info else ""
                print(f"  {i}. [{node.id}] {node_title} ({node_type}){conn_str}")
            else:
                print(f"  {i}. [{node.id}] {node_title} ({node_type})")

    def do_detail(self, args: str):
        """Show detailed information for a specific node

        Usage: detail --node <node_id>

        Arguments:
            --node <node_id>: ID of the node to show details for
        """
        parser = argparse.ArgumentParser(description="Show detailed information for a specific node", prog='detail')
        parser.add_argument('--node', required=True, help='ID of the node to show details for')

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        try:
            node = self.ai_action.dsl_file.get_node(parsed_args.node)
            if node is None:
                print(f"✗ Node not found: {parsed_args.node}")
                return

            print(f"\n=== Node Detail: {parsed_args.node} ===")
            print(f"Title: {node.data.title}")
            print(f"Type: {node.data.type}")
            print(f"\nData (JSON):")
            # Use model_dump to get the dictionary, then format as JSON
            data_dict = node.data.model_dump()
            json_str = json.dumps(data_dict, indent=2, ensure_ascii=False)
            print(json_str)
        except Exception as e:
            print(f"✗ Failed to get node details: {e}")

    def do_generate(self, args: str):
        """Generate and add a new node using AI

        Usage: generate --after <node_id> --type <node_type> [--title <title>] [-m <message>]

        Arguments:
            --after <node_id>: ID of the node after which to add the new node
            --type <node_type>: Type of node to generate (e.g., llm, code, http-request)
            --title <title>: Optional custom title for the new node
            -m, --message <message>: Custom intent/instruction for node generation
        """
        parser = argparse.ArgumentParser(description="Generate and add a new node using AI", prog='generate')
        parser.add_argument('--after', required=True, help='ID of the node after which to add the new node')
        parser.add_argument('--type', required=True, help='Type of node to generate')
        parser.add_argument('--title', help='Optional custom title for the new node')
        parser.add_argument('-m', '--message', help='Custom intent/instruction for node generation')

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        if not self.ai_action:
            print("✗ No workflow loaded")
            return

        try:
            print(f"\n🤖 Generating {parsed_args.type} node after {parsed_args.after}...")
            if parsed_args.message:
                print(f"   Intent: {parsed_args.message}")
            node_id = self.ai_action.generate_node(
                parsed_args.after,
                NodeType(parsed_args.type),
                user_message=parsed_args.message
            )
            print(f"✓ Added node: {node_id}")
        except Exception as e:
            print(f"✗ Generation failed: {e}")

    def do_quit(self, args: str):
        """Quit the CLI"""
        print("Goodbye!")
        return True

    def do_exit(self, args: str):
        """Exit the CLI (alias for quit)"""
        return self.do_quit(args)

    def do_EOF(self, args: str):
        """Handle Ctrl+D"""
        print("\nGoodbye!")
        return True

    def emptyline(self):
        """Don't repeat the last command on empty line"""
        pass

    def do_validate_resources(self, args: str):
        """Validate all DSL files in resources directory

        Usage: validate_resources [--dir <directory>]

        Options:
            --dir <directory>: Custom directory to validate (default: resources/Awesome-Dify-Workflow/DSL)
        """
        parser = argparse.ArgumentParser(description="Validate all DSL files in resources", prog='validate_resources')
        parser.add_argument('--dir', help='Custom directory to validate',
                          default=str(project_root / 'resources' / 'Awesome-Dify-Workflow' / 'DSL'))

        parsed_args = self._parse_args(parser, args)
        if not parsed_args:
            return

        dsl_dir = Path(parsed_args.dir)
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
            rel = yf.relative_to(project_root) if yf.is_relative_to(project_root) else yf
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
        cli.do_validate_resources('')
        return
    if args.file:
        cli.do_load(f'"{args.file}"')  # Use quotes to handle paths with spaces
    # Enter interactive mode
    cli.cmdloop()


if __name__ == '__main__':
    main()
