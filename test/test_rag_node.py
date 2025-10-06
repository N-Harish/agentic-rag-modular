import pytest
from unittest.mock import Mock, patch, MagicMock
from src.nodes.rag_node import get_retriever, rag_tool_node
from src.nodes.schema import AgentState
from langchain_core.documents import Document


class TestGetRetriever:
    """Test suite for get_retriever function"""
    
    @patch('src.nodes.rag_node.vs')
    def test_get_retriever_returns_retriever(self, mock_vs):
        """Test get_retriever returns retriever from vectorstore"""
        mock_retriever = Mock()
        mock_vs.get_retriever.return_value = mock_retriever
        
        result = get_retriever()
        
        mock_vs.get_retriever.assert_called_once_with(top_k=10)
        assert result == mock_retriever
    
    @patch('src.nodes.rag_node.vs')
    def test_get_retriever_default_top_k(self, mock_vs):
        """Test get_retriever uses default top_k=10"""
        mock_retriever = Mock()
        mock_vs.get_retriever.return_value = mock_retriever
        
        get_retriever()
        
        call_kwargs = mock_vs.get_retriever.call_args[1]
        assert call_kwargs['top_k'] == 10
    
    @patch('src.nodes.rag_node.vs')
    def test_get_retriever_multiple_calls(self, mock_vs):
        """Test multiple calls to get_retriever"""
        mock_retriever = Mock()
        mock_vs.get_retriever.return_value = mock_retriever
        
        result1 = get_retriever()
        result2 = get_retriever()
        
        assert mock_vs.get_retriever.call_count == 2
        assert result1 == result2


class TestRagToolNode:
    """Test suite for rag_tool_node function"""
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_basic(self, mock_get_retriever):
        """Test rag_tool_node retrieves and formats documents"""
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="Content 1", metadata={"page": 1}),
            Document(page_content="Content 2", metadata={"page": 2}),
            Document(page_content="Content 3", metadata={"page": 3})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="What is AI?",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        expected_context = "Content 1\n\nContent 2\n\nContent 3"
        assert result["pdf_context"] == expected_context
        assert result["query"] == "What is AI?"
        assert result["intent"] == "pdf"
        mock_retriever.invoke.assert_called_once_with("What is AI?")
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_empty_results(self, mock_get_retriever):
        """Test rag_tool_node with no documents returned"""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Unknown query",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        assert result["pdf_context"] == ""
        assert result["query"] == "Unknown query"
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_single_document(self, mock_get_retriever):
        """Test rag_tool_node with single document"""
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="Single content", metadata={})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Test query",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        assert result["pdf_context"] == "Single content"
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_preserves_state(self, mock_get_retriever):
        """Test rag_tool_node preserves other state fields"""
        mock_retriever = Mock()
        mock_docs = [Document(page_content="Content", metadata={})]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Test query",
            intent="pdf",
            pdf_context="Old context",
            final_response="Previous response",
            weather_data={"temp": 25}
        )
        
        result = rag_tool_node(input_state)
        
        assert result["query"] == "Test query"
        assert result["intent"] == "pdf"
        assert result["final_response"] == "Previous response"
        assert result["weather_data"] == {"temp": 25}
        assert result["pdf_context"] == "Content"  # Updated
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_formats_context_with_double_newlines(self, mock_get_retriever):
        """Test that rag_tool_node joins documents with double newlines"""
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="First", metadata={}),
            Document(page_content="Second", metadata={}),
            Document(page_content="Third", metadata={})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Query",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        assert "\n\n" in result["pdf_context"]
        assert result["pdf_context"].count("\n\n") == 2
        assert result["pdf_context"] == "First\n\nSecond\n\nThird"
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_different_queries(self, mock_get_retriever):
        """Test rag_tool_node with different query types"""
        mock_retriever = Mock()
        mock_docs = [Document(page_content="Result", metadata={})]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        queries = [
            "What is machine learning?",
            "Explain quantum computing",
            "Define artificial intelligence"
        ]
        
        for query in queries:
            input_state = AgentState(
                query=query,
                intent="pdf",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = rag_tool_node(input_state)
            
            mock_retriever.invoke.assert_called_with(query)
            assert result["query"] == query
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_invokes_retriever_correctly(self, mock_get_retriever):
        """Test that rag_tool_node invokes retriever with query"""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever
        
        test_query = "Specific test query"
        input_state = AgentState(
            query=test_query,
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        rag_tool_node(input_state)
        
        mock_retriever.invoke.assert_called_once_with(test_query)
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_handles_documents_with_metadata(self, mock_get_retriever):
        """Test rag_tool_node handles documents with metadata correctly"""
        mock_retriever = Mock()
        mock_docs = [
            Document(
                page_content="Important document content",
                metadata={"page": 1, "source": "doc.pdf", "score": 0.95}
            )
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Query",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        # Should only extract page_content, not metadata
        assert result["pdf_context"] == "Important document content"
        assert "metadata" not in result["pdf_context"]
    
    @patch('src.nodes.rag_node.get_retriever')
    def test_rag_tool_node_with_long_content(self, mock_get_retriever):
        """Test rag_tool_node with long document content"""
        mock_retriever = Mock()
        long_content = "A" * 1000
        mock_docs = [
            Document(page_content=long_content, metadata={})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        input_state = AgentState(
            query="Query",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = rag_tool_node(input_state)
        
        assert len(result["pdf_context"]) == 1000
        assert result["pdf_context"] == long_content


class TestRagNodeIntegration:
    """Integration tests for RAG node components"""
    
    @patch('src.nodes.rag_node.vs')
    def test_get_retriever_and_rag_tool_node_integration(self, mock_vs):
        """Test integration between get_retriever and rag_tool_node"""
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="Integrated content", metadata={})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_vs.get_retriever.return_value = mock_retriever
        
        # First get retriever
        retriever = get_retriever()
        
        # Then use in rag_tool_node
        input_state = AgentState(
            query="Integration test",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        with patch('src.nodes.rag_node.get_retriever', return_value=retriever):
            result = rag_tool_node(input_state)
        
        assert result["pdf_context"] == "Integrated content"
        mock_vs.get_retriever.assert_called_once_with(top_k=10)


# Fixtures for common test setup
@pytest.fixture
def sample_documents():
    """Fixture providing sample Document objects"""
    return [
        Document(page_content="Document 1 content", metadata={"page": 1}),
        Document(page_content="Document 2 content", metadata={"page": 2}),
        Document(page_content="Document 3 content", metadata={"page": 3})
    ]


@pytest.fixture
def mock_retriever_with_docs(sample_documents):
    """Fixture providing a mock retriever with sample documents"""
    retriever = Mock()
    retriever.invoke.return_value = sample_documents
    return retriever


@pytest.fixture
def rag_query_state():
    """Fixture for RAG query state"""
    return AgentState(
        query="What is artificial intelligence?",
        intent="pdf",
        pdf_context="",
        final_response="",
        weather_data=None
    )
