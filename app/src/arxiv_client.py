"""
ArXiv Client for searching and downloading research papers.
"""

import requests
import os
import feedparser
from urllib.parse import quote_plus
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ArXivClient:
    """Client for interacting with ArXiv API."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        try:
            query_wrapped = f'"{query}"'
            encoded_query = quote_plus(query_wrapped)
            url = f"{self.base_url}?search_query=all:{encoded_query}&start=0&max_results={max_results}"
            
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
            
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def download_pdf(self, arxiv_id: str, save_dir: str = "downloads") -> Optional[str]:
        """
        Download PDF from ArXiv.
        
        Args:
            arxiv_id: ArXiv ID of the paper
            save_dir: Directory to save the PDF
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
            
            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded PDF: {save_path}")
                return save_path
            else:
                logger.error(f"Failed to download {arxiv_id}. Status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading PDF {arxiv_id}: {e}")
            return None
    
    def download_papers(self, arxiv_ids: List[str], save_dir: str = "downloads") -> Dict[str, str]:
        """
        Download multiple papers.
        
        Args:
            arxiv_ids: List of ArXiv IDs
            save_dir: Directory to save PDFs
            
        Returns:
            Dictionary mapping ArXiv IDs to PDF paths
        """
        downloaded = {}
        
        for arxiv_id in arxiv_ids:
            pdf_path = self.download_pdf(arxiv_id, save_dir)
            if pdf_path:
                downloaded[arxiv_id] = pdf_path
        
        logger.info(f"Downloaded {len(downloaded)} out of {len(arxiv_ids)} papers")
        return downloaded 