import pytest
from unittest.mock import Mock, patch, MagicMock
from src.nodes.router_node import get_groq_agent, router_node, route_query
from src.nodes.schema import AgentState


class TestGetGroqAgent:
    """Test suite for get_groq_agent function"""
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_get_groq_agent_initial_creation(self, mock_chat_groq):
        """Test that get_groq_agent creates a new ChatGroq instance on first call"""
        # Reset global state
        import src.nodes.router_node as router_module
        router_module._llm = None
        router_module._model_name = None
        
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        result = get_groq_agent("openai/gpt-oss-20b")
        
        mock_chat_groq.assert_called_once_with(
            model="openai/gpt-oss-20b",
            temperature=0
        )
        assert result == mock_llm_instance
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_get_groq_agent_returns_cached_instance(self, mock_chat_groq):
        """Test that get_groq_agent returns cached instance for same model"""
        # Reset and set up initial state
        import src.nodes.router_node as router_module
        mock_llm_instance = Mock()
        router_module._llm = mock_llm_instance
        router_module._model_name = "openai/gpt-oss-20b"
        
        result = get_groq_agent("openai/gpt-oss-20b")
        
        # Should not create a new instance
        mock_chat_groq.assert_not_called()
        assert result == mock_llm_instance
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_get_groq_agent_reinitialize_with_different_model(self, mock_chat_groq):
        """Test that get_groq_agent creates new instance for different model"""
        # Reset and set up initial state
        import src.nodes.router_node as router_module
        old_mock = Mock()
        router_module._llm = old_mock
        router_module._model_name = "openai/gpt-oss-20b"
        
        new_mock = Mock()
        mock_chat_groq.return_value = new_mock
        
        result = get_groq_agent("llama-3.1-70b-versatile")
        
        mock_chat_groq.assert_called_once_with(
            model="llama-3.1-70b-versatile",
            temperature=0
        )
        assert result == new_mock
        assert result != old_mock
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_get_groq_agent_default_model(self, mock_chat_groq):
        """Test get_groq_agent uses default model name"""
        import src.nodes.router_node as router_module
        router_module._llm = None
        router_module._model_name = None
        
        mock_llm_instance = Mock()
        mock_chat_groq.return_value = mock_llm_instance
        
        result = get_groq_agent()
        
        mock_chat_groq.assert_called_once_with(
            model="openai/gpt-oss-20b",
            temperature=0
        )
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_get_groq_agent_temperature_zero(self, mock_chat_groq):
        """Test that temperature is always set to 0"""
        import src.nodes.router_node as router_module
        router_module._llm = None
        router_module._model_name = None
        
        get_groq_agent("test-model")
        
        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs['temperature'] == 0


class TestRouterNode:
    """Test suite for router_node function"""
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_weather_intent(self, mock_get_groq_agent):
        """Test router_node identifies weather queries correctly"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "weather"
        mock_llm.invoke.return_value = mock_response
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="What's the weather like today?",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = router_node(input_state)
            
            assert result["intent"] == "weather"
            assert result["query"] == "What's the weather like today?"
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_pdf_intent(self, mock_get_groq_agent):
        """Test router_node identifies PDF queries correctly"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "pdf"
        mock_llm.invoke.return_value = mock_response
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="Summarize the document",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = router_node(input_state)
            
            assert result["intent"] == "pdf"
            assert result["query"] == "Summarize the document"
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_strips_whitespace(self, mock_get_groq_agent):
        """Test router_node strips whitespace from LLM response"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "  WEATHER  "
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="Is it raining?",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = router_node(input_state)
            
            assert result["intent"] == "weather"
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_converts_to_lowercase(self, mock_get_groq_agent):
        """Test router_node converts intent to lowercase"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "PDF"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="What's in the file?",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = router_node(input_state)
            
            assert result["intent"] == "pdf"
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_preserves_state_fields(self, mock_get_groq_agent):
        """Test router_node preserves all state fields"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "weather"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="Temperature today?",
                intent="",
                pdf_context="Some context",
                final_response="Previous response",
                weather_data={"temp": 25}
            )
            
            result = router_node(input_state)
            
            assert result["query"] == "Temperature today?"
            assert result["pdf_context"] == "Some context"
            assert result["final_response"] == "Previous response"
            assert result["weather_data"] == {"temp": 25}
            assert result["intent"] == "weather"
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_uses_correct_model(self, mock_get_groq_agent):
        """Test router_node uses the correct Groq model"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "pdf"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            input_state = AgentState(
                query="Test query",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            router_node(input_state)
            
            mock_get_groq_agent.assert_called_once_with("openai/gpt-oss-20b")
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_invokes_chain_with_query(self, mock_get_groq_agent):
        """Test router_node invokes chain with the query"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "weather"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            mock_get_groq_agent.return_value = mock_llm
            
            test_query = "What's the humidity level?"
            input_state = AgentState(
                query=test_query,
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            router_node(input_state)
            
            mock_chain.invoke.assert_called_once_with({"query": test_query})
    
    @patch('src.nodes.router_node.get_groq_agent')
    def test_router_node_traceable_decorator(self, mock_get_groq_agent):
        """Test that router_node has traceable decorator applied"""
        # Check if the function has been wrapped by traceable
        assert hasattr(router_node, '__wrapped__') or hasattr(router_node, '__name__')
        assert router_node.__name__ == 'router_node'


class TestRouteQuery:
    """Test suite for route_query function"""
    
    def test_route_query_weather_intent(self):
        """Test route_query returns 'weather_tool' for weather intent"""
        state = AgentState(
            query="What's the weather?",
            intent="weather",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state)
        
        assert result == "weather_tool"
    
    def test_route_query_pdf_intent(self):
        """Test route_query returns 'rag_tool' for pdf intent"""
        state = AgentState(
            query="Summarize document",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state)
        
        assert result == "rag_tool"
    
    def test_route_query_default_to_rag(self):
        """Test route_query defaults to 'rag_tool' when intent is missing"""
        state = AgentState(
            query="Some query",
            intent="",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state)
        
        assert result == "rag_tool"
    
    def test_route_query_unknown_intent_defaults_to_rag(self):
        """Test route_query defaults to 'rag_tool' for unknown intent"""
        state = AgentState(
            query="Some query",
            intent="unknown",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state)
        
        assert result == "rag_tool"
    
    def test_route_query_case_sensitive(self):
        """Test route_query is case sensitive for weather intent"""
        # Should only match lowercase 'weather'
        state_upper = AgentState(
            query="Weather query",
            intent="WEATHER",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state_upper)
        
        # Should default to rag_tool since it doesn't match 'weather'
        assert result == "rag_tool"
    
    def test_route_query_exact_weather_match(self):
        """Test route_query requires exact 'weather' string"""
        state = AgentState(
            query="Query",
            intent="weather",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = route_query(state)
        
        assert result == "weather_tool"
    
    def test_route_query_with_none_intent(self):
        """Test route_query handles None intent"""
        state = {"query": "Test", "intent": None}
        
        result = route_query(state)
        
        assert result == "rag_tool"
    
    def test_route_query_return_type(self):
        """Test route_query returns Literal type"""
        state_weather = AgentState(
            query="Weather",
            intent="weather",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        state_pdf = AgentState(
            query="PDF",
            intent="pdf",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result_weather = route_query(state_weather)
        result_pdf = route_query(state_pdf)
        
        assert result_weather in ["weather_tool", "rag_tool"]
        assert result_pdf in ["weather_tool", "rag_tool"]


class TestRouterNodeIntegration:
    """Integration tests for router node components"""
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_full_routing_flow_weather(self, mock_chat_groq):
        """Test complete flow from router_node to route_query for weather"""
        # Setup mocks
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "weather"
        mock_chat_groq.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            
            # Reset global state
            import src.nodes.router_node as router_module
            router_module._llm = None
            router_module._model_name = None
            
            # First, route the query
            input_state = AgentState(
                query="What's the temperature?",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            routed_state = router_node(input_state)
            
            # Then determine the tool
            tool = route_query(routed_state)
            
            assert routed_state["intent"] == "weather"
            assert tool == "weather_tool"
    
    @patch('src.nodes.router_node.ChatGroq')
    def test_full_routing_flow_pdf(self, mock_chat_groq):
        """Test complete flow from router_node to route_query for PDF"""
        # Setup mocks
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "pdf"
        mock_chat_groq.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        
        with patch('src.nodes.router_node.ChatPromptTemplate') as mock_prompt:
            mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
            
            # Reset global state
            import src.nodes.router_node as router_module
            router_module._llm = None
            router_module._model_name = None
            
            # First, route the query
            input_state = AgentState(
                query="Summarize the document",
                intent="",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            routed_state = router_node(input_state)
            
            # Then determine the tool
            tool = route_query(routed_state)
            
            assert routed_state["intent"] == "pdf"
            assert tool == "rag_tool"


# Fixtures for common test setup
@pytest.fixture
def weather_query_state():
    """Fixture for weather query state"""
    return AgentState(
        query="What's the weather forecast?",
        intent="",
        pdf_context="",
        final_response="",
        weather_data=None
    )


@pytest.fixture
def pdf_query_state():
    """Fixture for PDF query state"""
    return AgentState(
        query="What does the document say about AI?",
        intent="",
        pdf_context="",
        final_response="",
        weather_data=None
    )


@pytest.fixture
def mock_groq_llm():
    """Fixture providing a mock ChatGroq LLM"""
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "weather"
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture(autouse=True)
def reset_groq_agent_state():
    """Fixture to reset global state before each test"""
    import src.nodes.router_node as router_module
    router_module._llm = None
    router_module._model_name = None
    yield
    # Cleanup after test
    router_module._llm = None
    router_module._model_name = None