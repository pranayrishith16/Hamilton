from doc_generator import DocumentationGenerator

# Initialize with your Hamilton project path
doc_gen = DocumentationGenerator("./")

# Scan the entire project for Python files
print("Scanning project...")
structure = doc_gen.scan_project()

# Generate markdown documentation
print("Generating documentation...")
markdown_docs = doc_gen.generate_markdown(structure)

# Save to file
with open("PROJECT_DOCS.md", "w") as f:
    f.write(markdown_docs)

print("✅ Documentation generated successfully!")
print(f"📄 Saved to: PROJECT_DOCS.md")
print(f"📊 Modules documented: {len(structure)}")
