#!/usr/bin/env python3
"""
Automated Documentation Generator for Python Projects

This script scans a Python project directory, extracts all functions and classes
with their docstrings, and generates comprehensive markdown documentation.

Usage:
    python doc_generator.py /path/to/project [output_file.md]
    
    # Example:
    python doc_generator.py ./hamilton PROJECT_DOCS.md
"""

import os
import sys
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class DocumentationGenerator:
    """Generate markdown documentation from Python project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")
        self.structure = {}
    
    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from AST node."""
        docstring = ast.get_docstring(node)
        return docstring if docstring else None
    
    def extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function signature and metadata."""
        args = []
        for arg in node.args.args:
            if arg.arg != 'self':
                args.append(arg.arg)
        
        # Extract type annotations
        return_annotation = ""
        if node.returns:
            return_annotation = ast.unparse(node.returns)
        
        return {
            "name": node.name,
            "args": args,
            "return_type": return_annotation,
            "docstring": self.extract_docstring(node),
            "type": "function",
            "is_private": node.name.startswith('_'),
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }
    
    def extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information including methods."""
        methods = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self.extract_function_info(item))
        
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            else:
                base_classes.append(ast.unparse(base))
        
        return {
            "name": node.name,
            "docstring": self.extract_docstring(node),
            "type": "class",
            "base_classes": base_classes,
            "methods": methods,
            "is_private": node.name.startswith('_')
        }
    
    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse Python file and extract public functions and classes."""
        items = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    items.append(self.extract_function_info(node))
                elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    items.append(self.extract_class_info(node))
        
        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")
        
        return items
    
    def scan_project(self, exclude_dirs: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Scan entire project and extract documentation."""
        if exclude_dirs is None:
            exclude_dirs = [
                '.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.venv',
                'node_modules', '.egg-info', 'build', 'dist', '.tox', '.eggs',
                '__pypackages__', 'htmlcov', '.coverage'
            ]
        
        file_structure = {}
        total_files = 0
        
        for py_file in sorted(self.project_root.rglob('*.py')):
            total_files += 1
            
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            
            relative_path = py_file.relative_to(self.project_root)
            module_path = str(relative_path).replace('/', '.').replace('.py', '')
            
            items = self.parse_file(py_file)
            if items:
                file_structure[module_path] = items
        
        print(f"✓ Scanned {total_files} Python files")
        print(f"✓ Found documentation in {len(file_structure)} modules")
        
        return file_structure
    
    def generate_markdown(self, file_structure: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate complete markdown documentation."""
        md_lines = [
            "# Project Documentation",
            "",
            "Auto-generated documentation from Python source code.",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Table of Contents",
            ""
        ]
        
        # Generate table of contents
        for module_path in sorted(file_structure.keys()):
            anchor = module_path.replace('.', '-').lower()
            md_lines.append(f"- [{module_path}](#{anchor})")
        
        md_lines.extend(["", "---", ""])
        
        # Generate detailed documentation
        for module_path in sorted(file_structure.keys()):
            items = file_structure[module_path]
            anchor = module_path.replace('.', '-').lower()
            
            # Module header
            md_lines.append(f"## {module_path}")
            md_lines.append("")
            
            # Process each item (function or class)
            for item in items:
                if item['type'] == 'function':
                    md_lines.extend(self._generate_function_docs(item))
                elif item['type'] == 'class':
                    md_lines.extend(self._generate_class_docs(item))
            
            md_lines.extend(["", "---", ""])
        
        return "\n".join(md_lines)
    
    def _generate_function_docs(self, func: Dict[str, Any]) -> List[str]:
        """Generate markdown documentation for a function."""
        lines = [f"### Function: `{func['name']}`"]
        lines.append("")
        
        # Signature
        args_str = ", ".join(func['args']) if func['args'] else ""
        return_str = f" -> {func['return_type']}" if func['return_type'] else ""
        prefix = "async " if func.get('is_async') else ""
        
        lines.append("```")
        lines.append(f"{prefix}def {func['name']}({args_str}){return_str}:")
        lines.append("```")
        lines.append("")
        
        # Docstring
        if func['docstring']:
            lines.append("**Description:**")
            lines.append("")
            lines.append(func['docstring'])
            lines.append("")
        else:
            lines.append("*No description provided.*")
            lines.append("")
        
        # Parameters
        if func['args']:
            lines.append("**Parameters:**")
            lines.append("")
            for arg in func['args']:
                lines.append(f"- `{arg}`")
            lines.append("")
        
        # Return type
        if func['return_type']:
            lines.append("**Returns:**")
            lines.append("")
            lines.append(f"- `{func['return_type']}`")
            lines.append("")
        
        return lines
    
    def _generate_class_docs(self, cls: Dict[str, Any]) -> List[str]:
        """Generate markdown documentation for a class."""
        lines = [f"### Class: `{cls['name']}`"]
        lines.append("")
        
        # Base classes
        if cls['base_classes']:
            bases = ", ".join([f"`{b}`" for b in cls['base_classes']])
            lines.append(f"**Inherits from:** {bases}")
            lines.append("")
        
        # Docstring
        if cls['docstring']:
            lines.append("**Description:**")
            lines.append("")
            lines.append(cls['docstring'])
            lines.append("")
        else:
            lines.append("*No description provided.*")
            lines.append("")
        
        # Methods
        if cls['methods']:
            lines.append("**Methods:**")
            lines.append("")
            for method in cls['methods']:
                args_str = ", ".join(method['args']) if method['args'] else ""
                method_sig = f"`{method['name']}({args_str})`"
                
                if method['docstring']:
                    first_line = method['docstring'].split('\n')[0]
                    lines.append(f"- {method_sig}: {first_line}")
                else:
                    lines.append(f"- {method_sig}")
            lines.append("")
        
        return lines


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python doc_generator.py <project_root> [output_file.md]")
        print("\nExample:")
        print("  python doc_generator.py ./hamilton PROJECT_DOCS.md")
        sys.exit(1)
    
    project_root = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "PROJECT_DOCS.md"
    
    print(f"📚 Generating documentation for: {project_root}")
    print()
    
    try:
        # Generate documentation
        doc_gen = DocumentationGenerator(project_root)
        structure = doc_gen.scan_project()
        markdown = doc_gen.generate_markdown(structure)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print()
        print(f"✅ Documentation saved to: {output_file}")
        print(f"📄 File size: {len(markdown):,} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
