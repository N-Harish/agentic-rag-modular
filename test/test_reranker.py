import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.reranker import CEReranker


class TestCEReranker:
    """Test suite for CEReranker class"""
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_default_parameters(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization with default parameters"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        
        reranker = CEReranker()
        
        mock_huggingface.assert_called_once_with(model_name="BAAI/bge-reranker-base")
        mock_cross_encoder_reranker.assert_called_once_with(model=mock_model, top_n=5)
        assert reranker.compressor == mock_compressor
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_custom_parameters(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization with custom parameters"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        
        custom_model = "custom/reranker-model"
        custom_top_n = 10
        
        reranker = CEReranker(model_name=custom_model, top_n=custom_top_n)
        
        mock_huggingface.assert_called_once_with(model_name=custom_model)
        mock_cross_encoder_reranker.assert_called_once_with(model=mock_model, top_n=custom_top_n)
        assert reranker.compressor == mock_compressor
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_with_top_n_zero(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization with top_n=0"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        
        reranker = CEReranker(top_n=0)
        
        mock_cross_encoder_reranker.assert_called_once_with(model=mock_model, top_n=0)
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_with_large_top_n(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization with large top_n value"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        
        reranker = CEReranker(top_n=100)
        
        mock_cross_encoder_reranker.assert_called_once_with(model=mock_model, top_n=100)
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    @patch('src.core.reranker.ContextualCompressionRetriever')
    def test_get_reranked_retriever_returns_compression_retriever(
        self, mock_contextual_compression, mock_cross_encoder_reranker, mock_huggingface
    ):
        """Test that get_reranked_retriever returns ContextualCompressionRetriever"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        mock_compression_retriever = Mock()
        mock_contextual_compression.return_value = mock_compression_retriever
        
        reranker = CEReranker()
        mock_base_retriever = Mock()
        
        result = reranker.get_reranked_retriever(mock_base_retriever)
        
        mock_contextual_compression.assert_called_once_with(
            base_compressor=mock_compressor,
            base_retriever=mock_base_retriever
        )
        assert result == mock_compression_retriever
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    @patch('src.core.reranker.ContextualCompressionRetriever')
    def test_get_reranked_retriever_with_custom_retriever(
        self, mock_contextual_compression, mock_cross_encoder_reranker, mock_huggingface
    ):
        """Test get_reranked_retriever with custom base retriever"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        
        reranker = CEReranker(model_name="custom/model", top_n=3)
        mock_base_retriever = Mock(spec=['get_relevant_documents'])
        mock_base_retriever.get_relevant_documents.return_value = []
        
        result = reranker.get_reranked_retriever(mock_base_retriever)
        
        mock_contextual_compression.assert_called_once_with(
            base_compressor=mock_compressor,
            base_retriever=mock_base_retriever
        )
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    @patch('src.core.reranker.ContextualCompressionRetriever')
    def test_get_reranked_retriever_multiple_calls(
        self, mock_contextual_compression, mock_cross_encoder_reranker, mock_huggingface
    ):
        """Test multiple calls to get_reranked_retriever with same reranker instance"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        
        reranker = CEReranker()
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        
        result1 = reranker.get_reranked_retriever(mock_retriever1)
        result2 = reranker.get_reranked_retriever(mock_retriever2)
        
        assert mock_contextual_compression.call_count == 2
        mock_contextual_compression.assert_any_call(
            base_compressor=mock_compressor,
            base_retriever=mock_retriever1
        )
        mock_contextual_compression.assert_any_call(
            base_compressor=mock_compressor,
            base_retriever=mock_retriever2
        )
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_compressor_accessible_after_init(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test that compressor is accessible as instance attribute"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_compressor.top_n = 5
        mock_cross_encoder_reranker.return_value = mock_compressor
        
        reranker = CEReranker()
        
        assert hasattr(reranker, 'compressor')
        assert reranker.compressor == mock_compressor
        assert reranker.compressor.top_n == 5
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_handles_model_loading_exception(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization when model loading fails"""
        mock_huggingface.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            CEReranker()
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    def test_init_handles_compressor_creation_exception(self, mock_cross_encoder_reranker, mock_huggingface):
        """Test CEReranker initialization when compressor creation fails"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_cross_encoder_reranker.side_effect = Exception("Compressor creation failed")
        
        with pytest.raises(Exception, match="Compressor creation failed"):
            CEReranker()
    
    @patch('src.core.reranker.HuggingFaceCrossEncoder')
    @patch('src.core.reranker.CrossEncoderReranker')
    @patch('src.core.reranker.ContextualCompressionRetriever')
    def test_get_reranked_retriever_handles_exception(
        self, mock_contextual_compression, mock_cross_encoder_reranker, mock_huggingface
    ):
        """Test get_reranked_retriever when ContextualCompressionRetriever creation fails"""
        mock_model = Mock()
        mock_huggingface.return_value = mock_model
        mock_compressor = Mock()
        mock_cross_encoder_reranker.return_value = mock_compressor
        mock_contextual_compression.side_effect = Exception("Compression retriever creation failed")
        
        reranker = CEReranker()
        mock_base_retriever = Mock()
        
        with pytest.raises(Exception, match="Compression retriever creation failed"):
            reranker.get_reranked_retriever(mock_base_retriever)
    
    def test_multiple_reranker_instances_independent(self):
        """Test that multiple CEReranker instances are independent"""
        with patch('src.core.reranker.HuggingFaceCrossEncoder') as mock_hf, \
             patch('src.core.reranker.CrossEncoderReranker') as mock_ce:
            
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_hf.side_effect = [mock_model1, mock_model2]
            
            mock_compressor1 = Mock()
            mock_compressor2 = Mock()
            mock_ce.side_effect = [mock_compressor1, mock_compressor2]
            
            reranker1 = CEReranker(top_n=3)
            reranker2 = CEReranker(top_n=7)
            
            assert reranker1.compressor == mock_compressor1
            assert reranker2.compressor == mock_compressor2
            assert reranker1.compressor != reranker2.compressor


# Fixtures for common test setup
@pytest.fixture
def mock_huggingface_model():
    """Fixture providing a mock HuggingFaceCrossEncoder model"""
    with patch('src.core.reranker.HuggingFaceCrossEncoder') as mock:
        model = Mock()
        mock.return_value = model
        yield mock


@pytest.fixture
def mock_compressor():
    """Fixture providing a mock CrossEncoderReranker compressor"""
    with patch('src.core.reranker.CrossEncoderReranker') as mock:
        compressor = Mock()
        mock.return_value = compressor
        yield mock


@pytest.fixture
def mock_base_retriever():
    """Fixture providing a mock BaseRetriever"""
    retriever = Mock()
    retriever.get_relevant_documents.return_value = [
        Mock(page_content="doc1", metadata={}),
        Mock(page_content="doc2", metadata={})
    ]
    return retriever