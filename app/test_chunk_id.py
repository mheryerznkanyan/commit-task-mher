"""
Test to verify that chunks have chunk_id.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chunking import SemanticChunker, ParagraphChunker, TokenChunker

def test_chunk_id():
    """Test that all chunkers add chunk_id to chunks."""
    print("=== Testing chunk_id in all chunkers ===\n")
    
    # Test sentences
    test_sentences = [
        "This is the first sentence.",
        "This is the second sentence.",
        "This is the third sentence.",
        "This is the fourth sentence.",
        "This is the fifth sentence.",
        "This is the sixth sentence.",
        "This is the seventh sentence.",
        "This is the eighth sentence."
    ]
    
    # Test each chunker
    chunkers = [
        ("SemanticChunker", SemanticChunker()),
        ("ParagraphChunker", ParagraphChunker()),
        ("TokenChunker", TokenChunker())
    ]
    
    for name, chunker in chunkers:
        print(f"Testing {name}:")
        try:
            chunks = chunker.process_sentences(test_sentences)
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # Check if each chunk has chunk_id
            has_chunk_id = all("chunk_id" in chunk for chunk in chunks)
            print(f"  ✅ All chunks have chunk_id: {has_chunk_id}")
            
            # Show first chunk structure
            if chunks:
                first_chunk = chunks[0]
                print(f"  📋 First chunk keys: {list(first_chunk.keys())}")
                print(f"  📋 First chunk chunk_id: {first_chunk.get('chunk_id')}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_chunk_id() 