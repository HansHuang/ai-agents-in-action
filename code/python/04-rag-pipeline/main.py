"""Entry point for the RAG pipeline module.

Run the interactive demo:
    python main.py

Or import individual components:
    from rag_pipeline import RAGPipeline
    from rag_evaluator import RAGEvaluator
    from advanced_retriever import AdvancedRetriever
    from rag_agent import RAGAgent
    from knowledge_base_manager import KnowledgeBaseManager
"""

from rag_pipeline import RAGPipeline  # noqa: F401
from rag_evaluator import RAGEvaluator  # noqa: F401
from advanced_retriever import AdvancedRetriever  # noqa: F401
from rag_agent import RAGAgent  # noqa: F401
from knowledge_base_manager import KnowledgeBaseManager  # noqa: F401

if __name__ == "__main__":
    from rag_pipeline import _run_demo
    _run_demo()
