# scripts/bundler.py
import os
import zipfile
from pathlib import Path
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResourceBundler")

PROMPT_DIR = "prompts"
CONFIG_DIR = "configs"


def bundle_package(output_zip:str=None):
    """
    Bundle all prompts and configs into a single ZIP archive.
    
    Args:
        output_zip: Output ZIP filename
    """
    try:
        resource_dir = "stephanie/resources"
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
        logger.info(f"Creating resource directory at {resource_dir}...")
        logger.info(f"Starting resource bundling. Writing to {resource_dir}/{output_zip}...")

        # Ensure output path is valid
        output_zip = output_zip or f"{resource_dir}/stephanie_assets.zip"
        output_path = Path(output_zip).resolve()
        logger.info(f"Output zip: {output_path}")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add prompts
            logger.info(f"Adding prompt files from {PROMPT_DIR}")
            for root, _, files in os.walk(PROMPT_DIR):
                for f in files:
                    arcname = os.path.join("resources", "prompts", os.path.relpath(root, PROMPT_DIR), f)
                    zipf.write(os.path.join(root, f), arcname)

            # Add configs
            logger.info(f"Adding config files from {CONFIG_DIR}")
            for root, _, files in os.walk(CONFIG_DIR):
                for f in files:
                    if f.endswith(".yaml") or f.endswith(".yml"):
                        arcname = os.path.join("resources", "configs", os.path.relpath(root, CONFIG_DIR), f)
                        zipf.write(os.path.join(root, f), arcname)

        logger.info(f"[+] Successfully created {output_zip}")
    except PermissionError as e:
        logger.error(f"[!] Permission denied when trying to write {output_zip}. Try running as admin or changing output path.")
    except Exception as e:
        logger.error(f"[!] Error during bundling: {e}", exc_info=True)


if __name__ == "__main__":
    bundle_package()