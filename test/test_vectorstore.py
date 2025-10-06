import pytest
from unittest.mock import Mock, patch, MagicMock, call
from src.core.vectorstore import VectorStoreQdrant
from langchain_core.documents import Document


class TestVectorStoreQdrant:
    """Test suite for VectorStoreQdrant class"""
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_init_default_parameters(self, mock_embeddings):
        """Test VectorStoreQdrant initialization with default parameters"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        url = "http://localhost:6333"
        vs = VectorStoreQdrant(url=url)
        
        assert vs.url == url
        assert vs.collection_name == "pdf-rag"
        assert vs.prefer_grpc is False
        assert vs.rerank is True
        mock_embeddings.assert_called_once_with(model="nomic-embed-text-v1.5")
        assert vs.embeddings == mock_embedding_instance
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_init_custom_parameters(self, mock_embeddings):
        """Test VectorStoreQdrant initialization with custom parameters"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        url = "http://custom-qdrant:6333"
        collection = "custom-collection"
        
        vs = VectorStoreQdrant(
            url=url,
            collection_name=collection,
            prefer_grpc=True,
            rerank=False
        )
        
        assert vs.url == url
        assert vs.collection_name == collection
        assert vs.prefer_grpc is True
        assert vs.rerank is False
        assert vs.embeddings == mock_embedding_instance
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_upsert_doc_creates_vectorstore(self, mock_qdrant_vs, mock_embeddings):
        """Test upsert_doc creates QdrantVectorStore from documents"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_instance = Mock()
        mock_qdrant_vs.from_documents.return_value = mock_qdrant_instance
        
        url = "http://localhost:6333"
        vs = VectorStoreQdrant(url=url, collection_name="test-collection")
        
        docs = [
            Document(page_content="doc1", metadata={"page": 1}),
            Document(page_content="doc2", metadata={"page": 2})
        ]
        
        result = vs.upsert_doc(docs)
        
        mock_qdrant_vs.from_documents.assert_called_once_with(
            docs,
            mock_embedding_instance,
            url=url,
            prefer_grpc=False,
            api_key='test-api-key',
            collection_name="test-collection"
        )
        assert result == mock_qdrant_instance
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_upsert_doc_with_prefer_grpc(self, mock_qdrant_vs, mock_embeddings):
        """Test upsert_doc with prefer_grpc=True"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_instance = Mock()
        mock_qdrant_vs.from_documents.return_value = mock_qdrant_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333", prefer_grpc=True)
        docs = [Document(page_content="test", metadata={})]
        
        vs.upsert_doc(docs)
        
        call_kwargs = mock_qdrant_vs.from_documents.call_args[1]
        assert call_kwargs['prefer_grpc'] is True
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {}, clear=True)
    def test_upsert_doc_without_api_key(self, mock_qdrant_vs, mock_embeddings):
        """Test upsert_doc when QDRANT_API_KEY is not set"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        docs = [Document(page_content="test", metadata={})]
        
        vs.upsert_doc(docs)
        
        call_kwargs = mock_qdrant_vs.from_documents.call_args[1]
        assert call_kwargs['api_key'] is None
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_get_existing_vectorestore(self, mock_qdrant_vs, mock_embeddings):
        """Test get_existing_vectorestore loads existing collection"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_instance = Mock()
        mock_qdrant_vs.from_existing_collection.return_value = mock_qdrant_instance
        
        url = "http://localhost:6333"
        collection = "existing-collection"
        vs = VectorStoreQdrant(url=url, collection_name=collection, prefer_grpc=True)
        
        with patch('builtins.print') as mock_print:
            result = vs.get_existing_vectorestore()
        
        mock_print.assert_called_once_with("Using QDrant URL :-", url)
        mock_qdrant_vs.from_existing_collection.assert_called_once_with(
            embedding=mock_embedding_instance,
            collection_name=collection,
            url=url,
            api_key='test-api-key',
            prefer_grpc=True
        )
        assert result == mock_qdrant_instance
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {}, clear=True)
    def test_get_existing_vectorestore_without_api_key(self, mock_qdrant_vs, mock_embeddings):
        """Test get_existing_vectorestore when QDRANT_API_KEY is not set"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        
        with patch('builtins.print'):
            vs.get_existing_vectorestore()
        
        call_kwargs = mock_qdrant_vs.from_existing_collection.call_args[1]
        assert call_kwargs['api_key'] is None
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_similarity_search(self, mock_embeddings):
        """Test similarity_search returns results from vectorstore"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        
        mock_vectorstore = Mock()
        mock_results = [
            Document(page_content="result1", metadata={"score": 0.9}),
            Document(page_content="result2", metadata={"score": 0.8})
        ]
        mock_vectorstore.similarity_search.return_value = mock_results
        
        query = "test query"
        results = vs.similarity_search(query, mock_vectorstore, top_k=10)
        
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=10)
        assert results == mock_results
        assert len(results) == 2
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_similarity_search_default_top_k(self, mock_embeddings):
        """Test similarity_search with default top_k value"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = []
        
        vs.similarity_search("query", mock_vectorstore)
        
        mock_vectorstore.similarity_search.assert_called_once_with("query", k=20)
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch('src.core.vectorstore.CEReranker')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_get_retriever_with_rerank(self, mock_reranker, mock_qdrant_vs, mock_embeddings):
        """Test get_retriever with reranking enabled"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_base_retriever = Mock()
        mock_qdrant_instance.as_retriever.return_value = mock_base_retriever
        mock_qdrant_vs.from_existing_collection.return_value = mock_qdrant_instance
        
        mock_reranker_instance = Mock()
        mock_reranked_retriever = Mock()
        mock_reranker_instance.get_reranked_retriever.return_value = mock_reranked_retriever
        mock_reranker.return_value = mock_reranker_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333", rerank=True)
        
        with patch('builtins.print'):
            result = vs.get_retriever(top_k=30, reranked_k=10)
        
        mock_reranker.assert_called_once_with(top_n=10)
        mock_qdrant_instance.as_retriever.assert_called_once_with(
            k=30,
            search_kwargs={'k': 30},
            search_type="mmr"
        )
        mock_reranker_instance.get_reranked_retriever.assert_called_once_with(mock_base_retriever)
        assert result == mock_reranked_retriever
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_get_retriever_without_rerank(self, mock_qdrant_vs, mock_embeddings):
        """Test get_retriever with reranking disabled"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_base_retriever = Mock()
        mock_qdrant_instance.as_retriever.return_value = mock_base_retriever
        mock_qdrant_vs.from_existing_collection.return_value = mock_qdrant_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333", rerank=False)
        
        with patch('builtins.print'):
            result = vs.get_retriever(top_k=30, reranked_k=10)
        
        mock_qdrant_instance.as_retriever.assert_called_once_with(
            k=10,
            search_kwargs={'k': 30},
            search_type="mmr"
        )
        assert result == mock_base_retriever
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch('src.core.vectorstore.CEReranker')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_get_retriever_default_parameters(self, mock_reranker, mock_qdrant_vs, mock_embeddings):
        """Test get_retriever with default top_k and reranked_k parameters"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        mock_qdrant_instance = Mock()
        mock_base_retriever = Mock()
        mock_qdrant_instance.as_retriever.return_value = mock_base_retriever
        mock_qdrant_vs.from_existing_collection.return_value = mock_qdrant_instance
        
        mock_reranker_instance = Mock()
        mock_reranker_instance.get_reranked_retriever.return_value = Mock()
        mock_reranker.return_value = mock_reranker_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333", rerank=True)
        
        with patch('builtins.print'):
            vs.get_retriever()
        
        mock_reranker.assert_called_once_with(top_n=5)
        mock_qdrant_instance.as_retriever.assert_called_once_with(
            k=20,
            search_kwargs={'k': 20},
            search_type="mmr"
        )
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_upsert_doc_empty_list(self, mock_qdrant_vs, mock_embeddings):
        """Test upsert_doc with empty document list"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_instance = Mock()
        mock_qdrant_vs.from_documents.return_value = mock_qdrant_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        docs = []
        
        result = vs.upsert_doc(docs)
        
        mock_qdrant_vs.from_documents.assert_called_once()
        assert result == mock_qdrant_instance
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_upsert_doc_handles_exception(self, mock_qdrant_vs, mock_embeddings):
        """Test upsert_doc handles exception from QdrantVectorStore"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_vs.from_documents.side_effect = Exception("Connection failed")
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        docs = [Document(page_content="test", metadata={})]
        
        with pytest.raises(Exception, match="Connection failed"):
            vs.upsert_doc(docs)
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    @patch('src.core.vectorstore.QdrantVectorStore')
    @patch.dict('os.environ', {'QDRANT_API_KEY': 'test-api-key'})
    def test_get_existing_vectorestore_handles_exception(self, mock_qdrant_vs, mock_embeddings):
        """Test get_existing_vectorestore handles exception"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        mock_qdrant_vs.from_existing_collection.side_effect = Exception("Collection not found")
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        
        with patch('builtins.print'):
            with pytest.raises(Exception, match="Collection not found"):
                vs.get_existing_vectorestore()
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_similarity_search_handles_exception(self, mock_embeddings):
        """Test similarity_search handles exception from vectorstore"""
        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance
        
        vs = VectorStoreQdrant(url="http://localhost:6333")
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception, match="Search failed"):
            vs.similarity_search("query", mock_vectorstore)
    
    @patch('src.core.vectorstore.NomicEmbeddings')
    def test_multiple_instances_independent(self, mock_embeddings):
        """Test that multiple VectorStoreQdrant instances are independent"""
        mock_embeddings.return_value = Mock()
        
        vs1 = VectorStoreQdrant(url="http://url1:6333", collection_name="col1", rerank=True)
        vs2 = VectorStoreQdrant(url="http://url2:6333", collection_name="col2", rerank=False)
        
        assert vs1.url == "http://url1:6333"
        assert vs2.url == "http://url2:6333"
        assert vs1.collection_name == "col1"
        assert vs2.collection_name == "col2"
        assert vs1.rerank is True
        assert vs2.rerank is False


# Fixtures for common test setup
@pytest.fixture
def mock_embeddings():
    """Fixture providing mock NomicEmbeddings"""
    with patch('src.core.vectorstore.NomicEmbeddings') as mock:
        embedding_instance = Mock()
        mock.return_value = embedding_instance
        yield mock


@pytest.fixture
def mock_qdrant_vectorstore():
    """Fixture providing mock QdrantVectorStore"""
    with patch('src.core.vectorstore.QdrantVectorStore') as mock:
        vectorstore_instance = Mock()
        mock.from_documents.return_value = vectorstore_instance
        mock.from_existing_collection.return_value = vectorstore_instance
        yield mock


@pytest.fixture
def sample_documents():
    """Fixture providing sample Document objects"""
    return [
        Document(page_content="Sample content 1", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="Sample content 2", metadata={"page": 2, "source": "test.pdf"}),
        Document(page_content="Sample content 3", metadata={"page": 3, "source": "test.pdf"})
    ]


@pytest.fixture
def vectorstore_instance():
    """Fixture providing a VectorStoreQdrant instance with mocked dependencies"""
    with patch('src.core.vectorstore.NomicEmbeddings'):
        vs = VectorStoreQdrant(url="http://localhost:6333", collection_name="test")
        yield vs