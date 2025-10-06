import pytest
from unittest.mock import Mock, patch, MagicMock
from src.nodes.weather_node import get_weather, weather_agent, weather_tool_node
from src.nodes.schema import AgentState


class TestGetWeatherTool:
    """Test suite for get_weather tool function"""
    
    @patch('src.nodes.weather_node.requests.get')
    @patch.dict('os.environ', {'OPENWEATHERMAP_API_KEY': 'test-api-key'})
    def test_get_weather_success(self, mock_requests_get):
        """Test get_weather returns weather data successfully"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "main": {"temp": 25.5, "humidity": 60},
            "weather": [{"description": "clear sky"}],
            "name": "Mumbai"
        }
        mock_requests_get.return_value = mock_response
        
        result = get_weather.invoke({"city": "Mumbai"})
        
        expected_url = "https://api.openweathermap.org/data/2.5/weather?q=Mumbai&appid=test-api-key&units=metric"
        mock_requests_get.assert_called_once_with(expected_url)
        assert result["name"] == "Mumbai"
        assert result["main"]["temp"] == 25.5
    
    @patch('src.nodes.weather_node.requests.get')
    @patch.dict('os.environ', {'OPENWEATHERMAP_API_KEY': 'test-api-key'})
    def test_get_weather_different_cities(self, mock_requests_get):
        """Test get_weather with different city names"""
        mock_response = Mock()
        mock_response.json.return_value = {"name": "London"}
        mock_requests_get.return_value = mock_response
        
        get_weather.invoke({"city": "London"})
        
        assert "London" in mock_requests_get.call_args[0][0]
    
    @patch('src.nodes.weather_node.requests.get')
    @patch.dict('os.environ', {'OPENWEATHERMAP_API_KEY': 'test-key'})
    def test_get_weather_api_key_in_url(self, mock_requests_get):
        """Test that API key is correctly included in URL"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response
        
        get_weather.invoke({"city": "Paris"})
        
        called_url = mock_requests_get.call_args[0][0]
        assert "appid=test-key" in called_url
        assert "units=metric" in called_url
    
    @patch('src.nodes.weather_node.requests.get')
    @patch.dict('os.environ', {}, clear=True)
    def test_get_weather_no_api_key(self, mock_requests_get):
        """Test get_weather when API key is not set"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response
        
        get_weather.invoke({"city": "Tokyo"})
        
        called_url = mock_requests_get.call_args[0][0]
        assert "appid=None" in called_url
    
    def test_get_weather_tool_metadata(self):
        """Test that get_weather tool has correct metadata"""
        assert get_weather.name == "get_weather"
        assert "weather" in get_weather.description.lower()
        assert callable(get_weather.func)


class TestWeatherAgent:
    """Test suite for weather_agent function"""
    
    @patch('src.nodes.weather_node.ChatGroq')
    @patch('src.nodes.weather_node.create_tool_calling_agent')
    @patch('src.nodes.weather_node.AgentExecutor')
    @patch('src.nodes.weather_node.ChatPromptTemplate')
    def test_weather_agent_default_model(
        self, mock_prompt_template, mock_agent_executor, mock_create_agent, mock_chatgroq
    ):
        """Test weather_agent with default model"""
        mock_llm = Mock()
        mock_chatgroq.return_value = mock_llm
        mock_llm.bind_tools.return_value = Mock()
        
        mock_prompt = Mock()
        mock_prompt_template.from_messages.return_value = mock_prompt
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        mock_executor = Mock()
        mock_agent_executor.return_value = mock_executor
        
        with patch('builtins.print'):
            result = weather_agent()
        
        mock_chatgroq.assert_called_once_with(
            model="openai/gpt-oss-20b",
            temperature=0.0
        )
        assert result == mock_executor
    
    @patch('src.nodes.weather_node.ChatGroq')
    @patch('src.nodes.weather_node.create_tool_calling_agent')
    @patch('src.nodes.weather_node.AgentExecutor')
    @patch('src.nodes.weather_node.ChatPromptTemplate')
    def test_weather_agent_custom_model(
        self, mock_prompt_template, mock_agent_executor, mock_create_agent, mock_chatgroq
    ):
        """Test weather_agent with custom model name"""
        mock_llm = Mock()
        mock_chatgroq.return_value = mock_llm
        mock_llm.bind_tools.return_value = Mock()
        
        mock_prompt_template.from_messages.return_value = Mock()
        mock_create_agent.return_value = Mock()
        mock_agent_executor.return_value = Mock()
        
        custom_model = "llama3-70b-8192"
        
        with patch('builtins.print'):
            weather_agent(groq_model_name=custom_model)
        
        mock_chatgroq.assert_called_once_with(
            model=custom_model,
            temperature=0.0
        )
    
    @patch('src.nodes.weather_node.ChatGroq')
    @patch('src.nodes.weather_node.create_tool_calling_agent')
    @patch('src.nodes.weather_node.AgentExecutor')
    @patch('src.nodes.weather_node.ChatPromptTemplate')
    def test_weather_agent_creates_agent_executor(
        self, mock_prompt_template, mock_agent_executor, mock_create_agent, mock_chatgroq
    ):
        """Test that weather_agent creates AgentExecutor with correct parameters"""
        mock_llm = Mock()
        mock_chatgroq.return_value = mock_llm
        mock_llm_with_tools = Mock()
        mock_llm.bind_tools.return_value = mock_llm_with_tools
        
        mock_prompt = Mock()
        mock_prompt_template.from_messages.return_value = mock_prompt
        
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        with patch('builtins.print'):
            weather_agent()
        
        # Verify create_tool_calling_agent was called
        assert mock_create_agent.call_count == 1
        
        # Verify AgentExecutor was called with correct parameters
        call_kwargs = mock_agent_executor.call_args[1]
        assert call_kwargs['agent'] == mock_agent
        assert call_kwargs['handle_parsing_errors'] is True
        assert call_kwargs['verbose'] is True
    
    # @patch('src.nodes.weather_node.ChatGroq')
    # @patch('src.nodes.weather_node.create_tool_calling_agent')
    # @patch('src.nodes.weather_node.AgentExecutor')
    # @patch('src.nodes.weather_node.ChatPromptTemplate')
    # def test_weather_agent_binds_tools_to_llm(
    #     self, mock_prompt_template, mock_agent_executor, mock_create_agent, mock_chatgroq
    # ):
    #     """Test that weather_agent binds tools to LLM"""
    #     mock_llm = Mock()
    #     mock_chatgroq.return_value = mock_llm
    #     mock_llm_with_tools = Mock()
    #     mock_llm.bind_tools.return_value = mock_llm_with_tools
        
    #     mock_prompt_template.from_messages.return_value = Mock()
    #     mock_create_agent.return_value = Mock()
    #     mock_agent_executor.return_value = Mock()
        
    #     with patch('builtins.print'):
    #         weather_agent()
        
    #     # Verify bind_tools was called
    #     assert mock_llm.bind_tools.call_count == 1
    #     called_tools = mock_llm.bind_tools.call_args[0][0]
    #     assert len(called_tools) == 1
    #     assert called_tools[0].name == "get_weather"
    
    @patch('src.nodes.weather_node.ChatGroq')
    @patch('src.nodes.weather_node.create_tool_calling_agent')
    @patch('src.nodes.weather_node.AgentExecutor')
    @patch('src.nodes.weather_node.ChatPromptTemplate')
    def test_weather_agent_prints_available_tools(
        self, mock_prompt_template, mock_agent_executor, mock_create_agent, mock_chatgroq
    ):
        """Test that weather_agent prints available tools"""
        mock_llm = Mock()
        mock_chatgroq.return_value = mock_llm
        mock_llm.bind_tools.return_value = Mock()
        
        mock_prompt_template.from_messages.return_value = Mock()
        mock_create_agent.return_value = Mock()
        mock_agent_executor.return_value = Mock()
        
        with patch('builtins.print') as mock_print:
            weather_agent()
        
        mock_print.assert_called_once()
        call_args = str(mock_print.call_args)
        assert "Available tools" in call_args


class TestWeatherToolNode:
    """Test suite for weather_tool_node function"""
    
    @patch('src.nodes.weather_node.weather_agent')
    def test_weather_tool_node_basic(self, mock_weather_agent):
        """Test weather_tool_node processes state correctly"""
        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "The weather in Mumbai is 25°C and clear"
        }
        mock_weather_agent.return_value = mock_executor
        
        input_state = AgentState(
            query="What's the weather in Mumbai?",
            intent="weather",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        result = weather_tool_node(input_state)
        
        assert result["query"] == "What's the weather in Mumbai?"
        assert result["intent"] == "weather"
        assert result["weather_data"] is not None
        mock_executor.invoke.assert_called_once_with(
            {"input": "What's the weather in Mumbai?"}
        )
    
    @patch('src.nodes.weather_node.weather_agent')
    def test_weather_tool_node_preserves_state(self, mock_weather_agent):
        """Test that weather_tool_node preserves other state fields"""
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Weather data"}
        mock_weather_agent.return_value = mock_executor
        
        input_state = AgentState(
            query="Weather query",
            intent="weather",
            pdf_context="Some PDF context",
            final_response="Previous response",
            weather_data=None
        )
        
        result = weather_tool_node(input_state)
        
        assert result["query"] == "Weather query"
        assert result["intent"] == "weather"
        assert result["pdf_context"] == "Some PDF context"
        assert result["final_response"] == "Previous response"
    
    @patch('src.nodes.weather_node.weather_agent')
    def test_weather_tool_node_invokes_agent(self, mock_weather_agent):
        """Test that weather_tool_node invokes agent executor"""
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Test output"}
        mock_weather_agent.return_value = mock_executor
        
        input_state = AgentState(
            query="Test query",
            intent="weather",
            pdf_context="",
            final_response="",
            weather_data=None
        )
        
        weather_tool_node(input_state)
        
        mock_weather_agent.assert_called_once()
        mock_executor.invoke.assert_called_once()
    
    @patch('src.nodes.weather_node.weather_agent')
    def test_weather_tool_node_with_different_queries(self, mock_weather_agent):
        """Test weather_tool_node with different query types"""
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Response"}
        mock_weather_agent.return_value = mock_executor
        
        queries = [
            "What's the temperature in London?",
            "Will it rain tomorrow in Paris?",
            "Current weather in Tokyo"
        ]
        
        for query in queries:
            input_state = AgentState(
                query=query,
                intent="weather",
                pdf_context="",
                final_response="",
                weather_data=None
            )
            
            result = weather_tool_node(input_state)
            
            assert result["query"] == query
            assert result["weather_data"] is not None


# Fixtures for common test setup
@pytest.fixture
def sample_agent_state():
    """Fixture providing a sample AgentState"""
    return AgentState(
        query="What's the weather in Mumbai?",
        intent="weather",
        pdf_context="",
        final_response="",
        weather_data=None
    )


@pytest.fixture
def mock_weather_response():
    """Fixture providing mock weather API response"""
    return {
        "coord": {"lon": 72.85, "lat": 19.01},
        "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
        "main": {
            "temp": 25.5,
            "feels_like": 26.2,
            "temp_min": 24.0,
            "temp_max": 27.0,
            "pressure": 1013,
            "humidity": 60
        },
        "wind": {"speed": 3.5},
        "name": "Mumbai"
    }


@pytest.fixture
def mock_agent_executor():
    """Fixture providing a mock AgentExecutor"""
    executor = Mock()
    executor.invoke.return_value = {"output": "Weather information"}
    return executor