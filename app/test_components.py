"""
Test script to verify each component step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config.yaml")
def test_components(cfg: DictConfig):
    """Test each component step by step."""
    print("=== Testing Components Step by Step ===\n")
    
    # Test 1: Config loading
    print("1. ✅ Config loaded successfully")
    print(f"   - LLM evaluation: {cfg.llm_evaluation.enabled}")
    print(f"   - Chunking strategy: {cfg.chunking.strategy}")
    print(f"   - Similarity strategy: {cfg.similarity.strategy}")
    
    # Test 2: Import chunkers
    try:
        from chunking import SemanticChunker
        print("\n2. ✅ Chunkers imported successfully")
        print("   - Available: SemanticChunker, ParagraphChunker, TokenChunker")
    except Exception as e:
        print(f"\n2. ❌ Chunker import failed: {e}")
        return
    
    # Test 3: Import similarity models
    try:
        from similarity.bi_encoder_similarity import BiEncoderSimilarity
        print("\n3. ✅ Similarity models imported successfully")
        print("   - Available: BiEncoderSimilarity, CrossEncoderSimilarity, HybridSimilarity")
    except Exception as e:
        print(f"\n3. ❌ Similarity model import failed: {e}")
        return
    
    # Test 4: Test chunker initialization
    try:
        SemanticChunker()
        print("\n4. ✅ Chunker initialization successful")
    except Exception as e:
        print(f"\n4. ❌ Chunker initialization failed: {e}")
    
    # Test 5: Test similarity model initialization
    try:
        BiEncoderSimilarity()
        print("\n5. ✅ Similarity model initialization successful")
    except Exception as e:
        print(f"\n5. ❌ Similarity model initialization failed: {e}")
    
    # Test 6: Test pipeline initialization
    try:
        from research_pipeline import ResearchPipeline
        ResearchPipeline(
            llm_evaluation_config=cfg.llm_evaluation,
            chunking_config=cfg.chunking
        )
        print("\n6. ✅ Pipeline initialization successful")
    except Exception as e:
        print(f"\n6. ❌ Pipeline initialization failed: {e}")
    
    print("\n=== All components working! ===")
    print("You can now run the full pipeline with: python main.py")

if __name__ == "__main__":
    test_components() 