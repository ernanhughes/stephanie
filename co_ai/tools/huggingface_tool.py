from huggingface_hub import HfApi
import re
import requests

def recommend_similar_papers(paper_url:str = "https://arxiv.org/pdf/2505.08827"):
    from gradio_client import Client
    client = Client("librarian-bots/recommend_similar_papers")
    result = client.predict(paper_url, None, False, api_name="/predict")
    print(result)
    paper_ids = re.findall(r"https://huggingface\.co/papers/(\d+\.\d+)", result)
    print(paper_ids)
    arxiv_pdf_urls = [f"https://arxiv.org/pdf/{pid}.pdf" for pid in paper_ids]
    print(arxiv_pdf_urls)
    for pid in paper_ids:
        url = f"https://arxiv.org/pdf/{pid}.pdf"
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{pid}.pdf", "wb") as f:
                f.write(response.content)
                print(f"Downloaded {pid}.pdf")
        else:
            print(f"Failed to download {pid}")
        return result

def search_huggingface_datasets(queries: list[str], max_results: int = 5) -> list[dict]:
    api = HfApi()
    results = []

    for query in queries:
        try:
            matches = api.list_datasets(search=query, limit=max_results)
            for ds in matches:
                results.append({
                    "name": ds.id,
                    "description": ds.cardData.get("description", "No description available") if ds.cardData else "No card data"
                })
        except Exception as e:
            results.append({
                "name": query,
                "description": f"Error searching: {str(e)}"
            })

    return results
