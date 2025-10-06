import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib

groq_module = importlib.import_module("src.nodes.final_response_node")
from src.nodes.final_response_node import get_groq_agent, generate_response_node
from src.nodes.schema import AgentState


class TestGetGroqAgent:
    @patch('src.nodes.final_response_node.ChatGroq')
    def test_lazy_initialization_and_reinit(self, mock_chatgroq):
        # First call with default model should construct ChatGroq
        mock_chatgroq.return_value = Mock(name='llm-default')
        llm1 = get_groq_agent()
        mock_chatgroq.assert_called_with(model="openai/gpt-oss-20b")
        assert llm1 is groq_module._llm

        # Calling again with same model should NOT call ChatGroq again
        mock_chatgroq.reset_mock()
        llm2 = get_groq_agent()
        mock_chatgroq.assert_not_called()
        assert llm1 is llm2

        # Calling with a different model should re-initialize
        mock_chatgroq.return_value = Mock(name='llm-custom')
        custom = "llama3-70b-8192"
        llm3 = get_groq_agent(groq_model_name=custom)
        mock_chatgroq.assert_called_once_with(model=custom)
        assert groq_module._model_name == custom
        assert llm3 is groq_module._llm


class TestGenerateResponseNodeWeather:
    @patch('src.nodes.final_response_node.get_groq_agent')
    @patch('src.nodes.final_response_node.ChatPromptTemplate')
    def test_weather_error_returns_error_message(self, mock_prompt_template, mock_get_agent):
        # Prepare a state where weather_data contains an error
        state = {
            "query": "What's the weather?",
            "intent": "weather",
            "weather_data": {"error": "api failure"},
            "pdf_context": "",
            "final_response": ""
        }

        # Call the function
        result = generate_response_node(state)

        # Some implementations mutate the input state and return None; handle both cases
        if result is None:
            result = state

        assert isinstance(result, dict)
        assert "final_response" in result
        assert "Error fetching weather data" in result['final_response']
        assert "api failure" in result['final_response']

    @patch('src.nodes.final_response_node.get_groq_agent')
    @patch('src.nodes.final_response_node.ChatPromptTemplate')
    def test_weather_success_invokes_prompt_chain(self, mock_prompt_template, mock_get_agent):
        # Setup mocks for llm and chaining
        mock_llm = Mock()
        mock_get_agent.return_value = mock_llm

        # Create a mock prompt whose __or__ returns a chain with an invoke method
        mock_prompt = MagicMock()
        mock_chain = MagicMock()
        mock_response = Mock()
        mock_response.content = "Sunny in Mumbai: 25°C, clear sky"
        mock_chain.invoke.return_value = mock_response
        mock_prompt.__or__.return_value = mock_chain

        # Ensure ChatPromptTemplate.from_messages returns our mock_prompt
        mock_prompt_template.from_messages.return_value = mock_prompt

        state = {
            "query": "What's the weather in Mumbai?",
            "intent": "weather",
            "weather_data": {"name": "Mumbai", "main": {"temp": 25}},
            "pdf_context": "",
            "final_response": ""
        }

        result = generate_response_node(state)

        # Some implementations mutate the input state and return None; handle both cases
        if result is None:
            result = state

        assert isinstance(result, dict)
        assert "final_response" in result
        assert result['final_response'] == mock_response.content
        # Validate that the chain was invoked with the provided inputs
        mock_chain.invoke.assert_called_once()


class TestGenerateResponseNodePDF:
    @patch('src.nodes.final_response_node.get_groq_agent')
    @patch('src.nodes.final_response_node.ChatPromptTemplate')
    @patch('src.nodes.final_response_node.SystemMessagePromptTemplate')
    @patch('src.nodes.final_response_node.HumanMessagePromptTemplate')
    @patch('src.nodes.final_response_node.StrOutputParser')
    def test_pdf_not_relevant_returns_insufficient_message(
        self, mock_str_parser, mock_human, mock_system, mock_chat_prompt, mock_get_agent
    ):
        # Setup llm stub
        mock_llm = Mock()
        mock_get_agent.return_value = mock_llm

        # Prepare the relevance prompt mock chain: relevance_prompt | llm | StrOutputParser()
        mock_relevance_prompt = MagicMock()
        mock_chain_stage1 = MagicMock()
        mock_chain_stage2 = MagicMock()

        # Chain stage2.invoke should return 'NOT_RELEVANT' (possibly with whitespace)
        mock_chain_stage2.invoke.return_value = ' NOT_RELEVANT '\
            .strip()  # simulate possible whitespace

        # Wire up the __or__ operator behaviour
        mock_relevance_prompt.__or__.return_value = mock_chain_stage1
        mock_chain_stage1.__or__.return_value = mock_chain_stage2

        # For the main ChatPromptTemplate.from_messages call(s), return the same mock
        mock_chat_prompt.from_messages.return_value = mock_relevance_prompt

        state = {
            "query": "Find data in PDF",
            "intent": "pdf",
            "weather_data": None,
            "pdf_context": "This PDF contains unrelated information",
            "final_response": ""
        }

        result = generate_response_node(state)

        # Some implementations mutate the input state and return None; handle both cases
        if result is None:
            result = state

        assert isinstance(result, dict)
        assert "final_response" in result
        assert "I don't have sufficient information" in result['final_response']

    @patch('src.nodes.final_response_node.get_groq_agent')
    @patch('src.nodes.final_response_node.ChatPromptTemplate')
    @patch('src.nodes.final_response_node.SystemMessagePromptTemplate')
    @patch('src.nodes.final_response_node.HumanMessagePromptTemplate')
    def test_pdf_relevant_returns_summary(self, mock_human, mock_system, mock_chat_prompt, mock_get_agent):
        mock_llm = Mock()
        mock_get_agent.return_value = mock_llm

        # First, the relevance chain should return 'RELEVANT'
        mock_relevance_prompt = MagicMock()
        chain1 = MagicMock()
        chain2 = MagicMock()
        chain2.invoke.return_value = 'RELEVANT'
        mock_relevance_prompt.__or__.return_value = chain1
        chain1.__or__.return_value = chain2

        # Second, the summarization prompt should be chained and return a response object with .content
        mock_summary_prompt = MagicMock()
        summary_chain = MagicMock()
        summary_response = Mock()
        summary_response.content = 'Concise summary based on context.'
        summary_chain.invoke.return_value = summary_response
        mock_summary_prompt.__or__.return_value = summary_chain

        # Make successive calls to ChatPromptTemplate.from_messages return relevance prompt then summary prompt
        mock_chat_prompt.from_messages.side_effect = [mock_relevance_prompt, mock_summary_prompt]

        state = {
            "query": "Summarize relevant PDF content",
            "intent": "pdf",
            "weather_data": None,
            "pdf_context": "This PDF contains the exact answer to the question.",
            "final_response": ""
        }

        result = generate_response_node(state)

        # Some implementations mutate the input state and return None; handle both cases
        if result is None:
            result = state

        assert isinstance(result, dict)
        assert "final_response" in result
        assert result['final_response'] == summary_response.content


# Helper fixture: ensure module-level cached llm reset between tests
@pytest.fixture(autouse=True)
def reset_module_state():
    groq_module._llm = None
    groq_module._model_name = None
    yield
