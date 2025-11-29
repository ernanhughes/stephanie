import os
import argparse
from pathlib import Path

# Config: Add any other folders you want to skip
EXCLUDE_DIRS = {
    '__pycache__', '.git', '.idea', 'venv', 'env', 
    'runs', 'data', 'logs', 'node_modules', 'dist', 'build', '.vscode'
}
# Config: Add extensions you want to capture
INCLUDE_EXTS = {'.py', '.yaml', '.yml', '.json', '.md', '.txt', '.toml', '.ini'}

def pack_project(root_dir=".", output_file="codebase_context.txt"):
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        print(f"❌ Error: Directory '{root_path}' does not exist.")
        return

    print(f"📦 Packing project from: {root_path}")
    print(f"📝 Output file: {output_file}")

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            # Write a header
            outfile.write(f"# Project Context: {root_path.name}\n")
            outfile.write(f"# Path: {root_path}\n")
            outfile.write("# Generated for AI Review\n\n")
            
            file_count = 0
            
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
                        # Relative path for cleaner context
                        rel_path = file_path.relative_to(root_path)
                        
                        # Format: Clear header for each file
                        outfile.write(f"\n{'='*50}\n")
                        outfile.write(f"FILE: {rel_path}\n")
                        outfile.write(f"{'='*50}\n\n")
                        outfile.write(content)
                        outfile.write("\n")
                        
                        print(f"  + Added: {rel_path}")
                        file_count += 1
                    except Exception as e:
                        print(f"  ! Skipped {file_path.name}: {e}")

        print(f"\n✅ Done! Packed {file_count} files into '{output_file}'.")
        print("   Upload this file to the chat.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pack a project codebase into a single text file for AI analysis.")
    
    # Argument: Root directory (optional, defaults to current dir)
    parser.add_argument(
        "path", 
        nargs="?", 
        default=".", 
        help="The path to the project directory you want to pack (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Run the packer with the provided path
    pack_project(root_dir=args.path)