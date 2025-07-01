# stephanie/utils/resource_extractor.py
import os
from pathlib import Path
from typing import List

import pkg_resources

RESOURCE_MAP = {
    "configs": "resources/configs",
    "prompts": "resources/prompts",
    "docker": "resources/docker"
}


def extract_resources(target_dir: str = None):
    """
    Extract all bundled configs and prompts into local directories.
    
    Args:
        target_dir: Optional override of where to extract files
    """
    target_dir = Path(target_dir or os.getcwd())
    
    for resource_type, source_path in RESOURCE_MAP.items():
        src = Path(source_path)
        dest = target_dir / resource_type
        
        # Skip if already exists
        if dest.exists():
            print(f"[ResourceExtractor] {resource_type} folder found. Skipping.")
            continue
            
        print(f"[ResourceExtractor] Copying {resource_type}...")
        dest.mkdir(exist_ok=True)

        # Walk through resource tree
        for root, _, files in os.walk(pkg_resources.resource_filename(__name__, src)):
            rel_root = Path(root).relative_to(pkg_resources.resource_filename(__name__, ""))
            for f in files:
                src_file = rel_root / f
                dst_file = dest / rel_root.relative_to(resource_type) / f
                
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(src_file, "rb") as fin:
                    content = fin.read().decode("utf-8")

                with open(dst_file, "w", encoding="utf-8") as fout:
                    fout.write(content)

    print("[ResourceExtractor] Done. Resources extracted to:")
    print(f" - Configs: {target_dir / 'configs'}")
    print(f" - Prompts: {target_dir / 'prompts'}")