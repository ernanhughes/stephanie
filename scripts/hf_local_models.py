# stephanie/utils/hf_tools.py
from huggingface_hub import scan_cache_dir
import humanize

def list_local_models(verbose: bool = True):
    """
    List all locally cached Hugging Face models.

    Args:
        verbose (bool): If True, print details. Otherwise, just return list of IDs.

    Returns:
        list[dict]: Metadata for each cached model
    """
    cache_info = scan_cache_dir()
    models = []

    for repo in cache_info.repos:
        local_path = getattr(repo, "local_dir", None) or getattr(repo, "repo_path", None)

        model_info = {
            "repo_id": repo.repo_id,
            "repo_type": repo.repo_type,
            "size_on_disk": humanize.naturalsize(repo.size_on_disk, binary=True),
            "local_path": str(local_path),
        }
        models.append(model_info)

        if verbose:
            print(f"\n📦 {repo.repo_id}")
            print(f"   • Type: {repo.repo_type}")
            print(f"   • Size: {model_info['size_on_disk']}")
            print(f"   • Path: {model_info['local_path']}")

    return models


if __name__ == "__main__":
    print("🔎 Scanning for local Hugging Face models...")
    list_local_models()
