import os

EXCLUDE_DIRS = {".git", ".github", ".venv", "orchestrations", "__pycache__"}
EXCLUDE_FILES = {".DS_Store"}

def list_directory_tree(root_dir, md_file='PROJECT_STRUCTURE.md'):
    """
    Recursively lists folder/files into markdown, skipping excluded folders/files.
    """
    lines = [f'# Project Directory Structure\n', f'Root: `{root_dir}`\n']
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to exclude unwanted directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        depth = dirpath[len(root_dir):].count(os.sep)
        indent = '  ' * depth
        folder = os.path.basename(dirpath) if os.path.basename(dirpath) else root_dir
        lines.append(f'{indent}- **{folder}/**')
        # Filter out unwanted files
        for fname in sorted([f for f in filenames if f not in EXCLUDE_FILES and not f.endswith('.pyc')]):
            lines.append(f'{indent}  - {fname}')
    with open(md_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Directory structure written to {md_file}')

if __name__ == "__main__":
    list_directory_tree(".", "PROJECT_STRUCTURE.md")
