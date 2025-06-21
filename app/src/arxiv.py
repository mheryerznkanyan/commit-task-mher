import requests
import os
import fitz


def download_arxiv_pdf(arxiv_id, save_dir="downloads"):
    """
    Downloads PDF from arXiv given an arXiv ID.
    """
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Build the URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # Build the file path
    save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    # Download the PDF
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
        return save_path
    else:
        print(f"Failed to download {arxiv_id}. Status code: {response.status_code}")
        return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz) for better text extraction.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    


def search_arxiv(query, max_results=5):
    """
    Search arXiv for papers matching the query.
    """
    base_url = "http://export.arxiv.org/api/query?"
    # URL encode the query
    query_wrapped = f'"{query}"' 

    # URL encode
    encoded_query = quote_plus(query_wrapped)
    print(encoded_query)
    print(query)
    # exit()
    # search_query = f"search_query=all:&start=0&max_results="
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"


    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split('/abs/')[-1]
        papers.append({
            'arxiv_id': arxiv_id,
            'title': entry.title,
            'summary': entry.summary,
            'link': entry.link
        })

    return papers

# Example usage:
results = search_arxiv("LLM models in 2025", max_results=3)
for paper in results:
    print(f"ID: {paper['arxiv_id']}")
    print(f"Title: {paper['title']}\n")
