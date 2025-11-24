import os
from pathlib import Path

# Config: Add any other folders you want to skip
EXCLUDE_DIRS = {
    '__pycache__', '.git', '.idea', 'venv', 'env', 
    'runs', 'data', 'logs', 'node_modules', 'dist'
}
# Config: Add extensions you want to capture
INCLUDE_EXTS = {'.py', '.yaml', '.yml', '.json', '.md'}

def pack_project(root_dir=".", output_file="codebase_context.txt"):
    root_path = Path(root_dir)
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Write a header
        outfile.write(f"# Project Context: {root_path.resolve().name}\n")
        outfile.write("# Generated for AI Review\n\n")
        
        for root, dirs, files in os.walk(root_path):
            # Modify dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                
                # Filter by extension
                if file_path.suffix not in INCLUDE_EXTS:
                    continue
                
                # Skip the output file itself and the packer script
                if file in [output_file, "pack_code.py", "package-lock.json"]:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    # Format: Clear header for each file
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"FILE: {file_path.relative_to(root_path)}\n")
                    outfile.write(f"{'='*50}\n\n")
                    outfile.write(content)
                    outfile.write("\n")
                    print(f"Packed: {file_path}")
                except Exception as e:
                    print(f"Skipped {file_path}: {e}")

    print(f"\n✅ Done! Upload '{output_file}' to the chat.")

if __name__ == "__main__":
    pack_project()