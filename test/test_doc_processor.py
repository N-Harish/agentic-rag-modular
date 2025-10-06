import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.doc_processor import DocProcessor


class TestDocProcessor:
    """Test suite for DocProcessor class"""
    
    def test_init_default_parameters(self):
        """Test DocProcessor initialization with default parameters"""
        processor = DocProcessor()
        
        assert processor.strategy == 'hi_res'
        assert processor.partition_via_api is True
        assert processor.coordinates is True
    
    def test_init_custom_parameters(self):
        """Test DocProcessor initialization with custom parameters"""
        processor = DocProcessor(
            strategy='fast',
            partition_via_api=False,
            coordinates=False
        )
        
        assert processor.strategy == 'fast'
        assert processor.partition_via_api is False
        assert processor.coordinates is False
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_load_pdf_splitter_creates_loader(self, mock_loader):
        """Test that load_pdf_splitter creates UnstructuredLoader with correct parameters"""
        processor = DocProcessor(
            strategy='hi_res',
            partition_via_api=True,
            coordinates=True
        )
        filepath = 'test.pdf'
        
        result = processor.load_pdf_splitter(filepath)
        
        mock_loader.assert_called_once_with(
            file_path=filepath,
            strategy='hi_res',
            partition_via_api=True,
            coordinates=True
        )
        assert result == mock_loader.return_value
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_load_pdf_splitter_with_custom_strategy(self, mock_loader):
        """Test load_pdf_splitter with custom strategy"""
        processor = DocProcessor(strategy='fast', partition_via_api=False)
        filepath = 'custom.pdf'
        
        processor.load_pdf_splitter(filepath)
        
        mock_loader.assert_called_once_with(
            file_path=filepath,
            strategy='fast',
            partition_via_api=False,
            coordinates=True
        )
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_returns_documents(self, mock_loader):
        """Test that process_pdf returns a list of documents"""
        # Setup mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "Document 1 content"
        mock_doc2 = Mock()
        mock_doc2.page_content = "Document 2 content"
        
        # Setup mock loader
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.return_value = iter([mock_doc1, mock_doc2])
        mock_loader.return_value = mock_loader_instance
        
        processor = DocProcessor()
        filepath = 'test.pdf'
        
        result = processor.process_pdf(filepath)
        
        assert len(result) == 2
        assert result[0] == mock_doc1
        assert result[1] == mock_doc2
        mock_loader_instance.lazy_load.assert_called_once()
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_empty_document(self, mock_loader):
        """Test process_pdf with empty document"""
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.return_value = iter([])
        mock_loader.return_value = mock_loader_instance
        
        processor = DocProcessor()
        result = processor.process_pdf('empty.pdf')
        
        assert result == []
        assert len(result) == 0
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_single_document(self, mock_loader):
        """Test process_pdf with single document"""
        mock_doc = Mock()
        mock_doc.page_content = "Single page content"
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.return_value = iter([mock_doc])
        mock_loader.return_value = mock_loader_instance
        
        processor = DocProcessor()
        result = processor.process_pdf('single.pdf')
        
        assert len(result) == 1
        assert result[0].page_content == "Single page content"
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_calls_load_pdf_splitter(self, mock_loader):
        """Test that process_pdf calls load_pdf_splitter internally"""
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.return_value = iter([])
        mock_loader.return_value = mock_loader_instance
        
        processor = DocProcessor(strategy='auto', partition_via_api=False)
        filepath = 'verify.pdf'
        
        processor.process_pdf(filepath)
        
        # Verify UnstructuredLoader was called with correct parameters
        mock_loader.assert_called_once_with(
            file_path=filepath,
            strategy='auto',
            partition_via_api=False,
            coordinates=True
        )
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_handles_loader_exception(self, mock_loader):
        """Test process_pdf behavior when loader raises exception"""
        mock_loader.side_effect = Exception("Failed to load PDF")
        
        processor = DocProcessor()
        
        with pytest.raises(Exception, match="Failed to load PDF"):
            processor.process_pdf('error.pdf')
    
    @patch('src.core.doc_processor.UnstructuredLoader')
    def test_process_pdf_handles_lazy_load_exception(self, mock_loader):
        """Test process_pdf behavior when lazy_load raises exception"""
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.side_effect = Exception("Lazy load failed")
        mock_loader.return_value = mock_loader_instance
        
        processor = DocProcessor()
        
        with pytest.raises(Exception, match="Lazy load failed"):
            processor.process_pdf('error.pdf')
    
    def test_multiple_processors_independent(self):
        """Test that multiple DocProcessor instances are independent"""
        processor1 = DocProcessor(strategy='hi_res')
        processor2 = DocProcessor(strategy='fast')
        
        assert processor1.strategy == 'hi_res'
        assert processor2.strategy == 'fast'
        assert processor1.strategy != processor2.strategy


# Fixtures for common test setup
@pytest.fixture
def default_processor():
    """Fixture providing a default DocProcessor instance"""
    return DocProcessor()


@pytest.fixture
def custom_processor():
    """Fixture providing a custom configured DocProcessor instance"""
    return DocProcessor(strategy='fast', partition_via_api=False, coordinates=False)


@pytest.fixture
def mock_documents():
    """Fixture providing mock document objects"""
    docs = []
    for i in range(3):
        doc = Mock()
        doc.page_content = f"Content of page {i+1}"
        doc.metadata = {'page': i+1}
        docs.append(doc)
    return docs
