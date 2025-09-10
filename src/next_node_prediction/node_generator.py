"""
AI-powered node generation using Claude API
Pure generation logic without validation
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)


class NodeGenerator:
    """Generate node data using Claude API"""
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-opus-4-1-20250805"
    
    def generate_node(self, 
                     node_type: str,
                     context: Dict[str, Any],
                     schema: Optional[Dict[str, Any]] = None,
                     previous_errors: Optional[list] = None) -> Dict[str, Any]:
        """
        Generate node data based on workflow context
        
        Args:
            node_type: Type of node to generate
            context: Workflow context from ContextBuilder
            schema: Optional JSON schema for the node type
            previous_errors: Optional list of previous validation errors for retry
        
        Returns:
            Generated node data (raw, unvalidated)
        """
        prompt = self._build_prompt(node_type, context, schema, previous_errors)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            
            if not content or not content.strip():
                raise ValueError("AI returned empty response")
            
            # Clean up any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Further cleanup - remove any leading/trailing text
            content = content.strip()
            
            # Find JSON object bounds
            start = content.find('{')
            end = content.rfind('}')
            
            if start == -1 or end == -1:
                raise ValueError(f"No valid JSON object found in response: {content[:200]}...")
            
            json_content = content[start:end+1]
            
            # Parse and return JSON
            return json.loads(json_content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse generated JSON: {e}\nContent: {content[:200] if 'content' in locals() else 'No content'}")
        except Exception as e:
            raise RuntimeError(f"AI generation failed: {e}")
    
    def _build_prompt(self, 
                     node_type: str,
                     context: Dict[str, Any],
                     schema: Optional[Dict[str, Any]],
                     previous_errors: Optional[list]) -> str:
        """Build generation prompt"""
        
        # Extract node data sequence
        node_sequence = []
        for node in context['node_sequence']:
            node_info = {
                'id': node['id'],
                'type': node['data'].get('type'),
                'title': node['data'].get('title')
            }
            
            # Add key outputs/variables
            if node['data'].get('type') == 'start':
                variables = node['data'].get('variables', [])
                node_info['outputs'] = [v.get('variable') for v in variables]
            elif node['data'].get('type') == 'code':
                node_info['outputs'] = list(node['data'].get('outputs', {}).keys())
            elif node['data'].get('type') == 'llm':
                node_info['outputs'] = ['text']  # LLM nodes output 'text'
            
            node_sequence.append(node_info)
        
        prompt = f"""Generate configuration for a {node_type} node in a Dify workflow.

## Workflow Context
App: {context['app_name']}
Description: {context['description']}

## Node Sequence
Previous nodes in the workflow:
{json.dumps(node_sequence, indent=2)}

## Available Variables
You can reference outputs from previous nodes using {{{{#node_id.variable#}}}} syntax.
Examples based on the nodes above:"""
        
        # Add variable resources
        for node in node_sequence:
            if node.get('outputs'):
                for output in node['outputs']:
                    prompt += f"\n- {{{{#{node['id']}.{output}#}}}}"
        
        # Add schema if provided
        if schema:
            prompt += f"\n\n## Node Schema\nThe {node_type} node should follow this structure:\n{json.dumps(self._simplify_schema(schema), indent=2)}"
        
        # Add error feedback if retrying
        if previous_errors:
            prompt += "\n\n## Previous Validation Errors\nThe previous attempt had these errors:\n"
            for error in previous_errors:
                prompt += f"- {error}\n"
            prompt += "\nPlease fix these issues in your response."
        
        # Add instructions
        prompt += f"""

## Instructions
1. Generate a valid 'data' field for a {node_type} node
2. Use appropriate variables from previous nodes
3. Ensure all required fields are included
4. Make the node's purpose clear and logical within the workflow

Return ONLY the JSON object for the data field, no explanation or markdown."""
        
        return prompt
    
    def _simplify_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify JSON schema for better AI understanding"""
        # Focus on required fields and common patterns
        simplified = {}
        
        # Extract required fields
        if 'required' in schema:
            simplified['required_fields'] = schema['required']
        
        # Extract property types
        if 'properties' in schema:
            simplified['fields'] = {}
            for field, field_schema in schema['properties'].items():
                field_type = field_schema.get('type', 'any')
                description = field_schema.get('description', '')
                
                simplified['fields'][field] = {
                    'type': field_type,
                    'description': description[:100] if description else 'No description'
                }
                
                # Add enum values if present
                if 'enum' in field_schema:
                    simplified['fields'][field]['allowed_values'] = field_schema['enum']
        
        return simplified