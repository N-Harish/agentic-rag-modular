import pytest
from unittest.mock import Mock, patch, MagicMock, call
from src.nodes.graph import create_graph
from src.nodes.schema import AgentState
from src.nodes.graph import invoke_runnable


def test_invoke_runnable_calls_agent_invoke_with_explicit_args():
    # Arrange
    agent = Mock()
    expected_result = {"ok": True}
    agent.invoke.return_value = expected_result

    prompt = "Check weather"
    intent = "weather"
    weather_data = {"temp": 25}
    pdf_text = "some pdf context"
    final_response = "Here you go"

    expected_payload = {
        "query": prompt,
        "intent": intent,
        "weather_data": weather_data,
        "pdf_context": pdf_text,     # note: invoke_runnable uses key "pdf_context"
        "final_response": final_response
    }

    # Act
    result = invoke_runnable(agent, prompt, intent=intent, weather_data=weather_data,
                             pdf_contexxt=pdf_text, final_response=final_response)

    # Assert
    agent.invoke.assert_called_once_with(expected_payload)
    assert result is expected_result


def test_invoke_runnable_uses_defaults_when_not_provided():
    # Arrange
    agent = Mock()
    expected_result = {"ok": "defaults"}
    agent.invoke.return_value = expected_result

    prompt = "Hello"

    expected_payload = {
        "query": prompt,
        "intent": "",
        "weather_data": {},          # default mutable dict in current implementation
        "pdf_context": "",          # default pdf_contexxt -> becomes pdf_context in payload
        "final_response": ""
    }

    # Act
    result = invoke_runnable(agent, prompt)

    # Assert
    agent.invoke.assert_called_once_with(expected_payload)
    assert result is expected_result


class TestCreateGraph:
    """Test suite for create_graph function"""
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_creates_state_graph(self, mock_state_graph):
        """Test that create_graph creates StateGraph with AgentState"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        mock_state_graph.assert_called_once_with(AgentState)
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_adds_all_nodes(self, mock_state_graph):
        """Test that create_graph adds all required nodes"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Check that add_node was called for each node
        expected_nodes = ["router", "weather_tool", "rag_tool", "generate_response"]
        assert mock_workflow.add_node.call_count == 4
        
        # Get all node names that were added
        added_nodes = [call[0][0] for call in mock_workflow.add_node.call_args_list]
        
        for node_name in expected_nodes:
            assert node_name in added_nodes
    
    @patch('src.nodes.graph.StateGraph')
    @patch('src.nodes.graph.router_node')
    @patch('src.nodes.graph.weather_tool_node')
    @patch('src.nodes.graph.rag_tool_node')
    @patch('src.nodes.graph.generate_response_node')
    def test_create_graph_adds_nodes_with_correct_functions(
        self, mock_gen_node, mock_rag_node, mock_weather_node, mock_router_node, mock_state_graph
    ):
        """Test that nodes are added with correct function references"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Verify add_node calls with correct functions
        mock_workflow.add_node.assert_any_call("router", mock_router_node)
        mock_workflow.add_node.assert_any_call("weather_tool", mock_weather_node)
        mock_workflow.add_node.assert_any_call("rag_tool", mock_rag_node)
        mock_workflow.add_node.assert_any_call("generate_response", mock_gen_node)
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_sets_entry_point(self, mock_state_graph):
        """Test that create_graph sets router as entry point"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        mock_workflow.set_entry_point.assert_called_once_with("router")
    
    @patch('src.nodes.graph.StateGraph')
    @patch('src.nodes.graph.route_query')
    def test_create_graph_adds_conditional_edges(self, mock_route_query, mock_state_graph):
        """Test that create_graph adds conditional edges from router"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Verify conditional edges from router
        mock_workflow.add_conditional_edges.assert_called_once_with(
            "router",
            mock_route_query,
            {
                "weather_tool": "weather_tool",
                "rag_tool": "rag_tool"
            }
        )
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_adds_edges_to_generate_response(self, mock_state_graph):
        """Test that create_graph adds edges from tools to generate_response"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Check edges to generate_response
        mock_workflow.add_edge.assert_any_call("weather_tool", "generate_response")
        mock_workflow.add_edge.assert_any_call("rag_tool", "generate_response")
    
    @patch('src.nodes.graph.StateGraph')
    @patch('src.nodes.graph.END')
    def test_create_graph_adds_edge_to_end(self, mock_end, mock_state_graph):
        """Test that create_graph adds edge from generate_response to END"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Check edge to END
        mock_workflow.add_edge.assert_any_call("generate_response", mock_end)
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_compiles_workflow(self, mock_state_graph):
        """Test that create_graph compiles the workflow"""
        mock_workflow = Mock()
        mock_compiled_app = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_compiled_app
        
        result = create_graph()
        
        mock_workflow.compile.assert_called_once()
        assert result == mock_compiled_app
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_returns_compiled_app(self, mock_state_graph):
        """Test that create_graph returns compiled application"""
        mock_workflow = Mock()
        mock_compiled_app = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_compiled_app
        
        result = create_graph()
        
        assert result is not None
        assert result == mock_compiled_app
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_correct_edge_count(self, mock_state_graph):
        """Test that create_graph adds correct number of edges"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Should have 3 regular edges:
        # weather_tool -> generate_response
        # rag_tool -> generate_response
        # generate_response -> END
        assert mock_workflow.add_edge.call_count == 3
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_correct_conditional_edge_count(self, mock_state_graph):
        """Test that create_graph adds correct number of conditional edges"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Should have 1 conditional edge from router
        assert mock_workflow.add_conditional_edges.call_count == 1


class TestGraphStructure:
    """Test suite for graph structure validation"""
    
    @patch('src.nodes.graph.StateGraph')
    def test_graph_has_correct_node_flow_weather(self, mock_state_graph):
        """Test graph structure for weather flow"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Verify the flow: router -> (conditional) -> weather_tool -> generate_response -> END
        # Router node exists
        node_calls = [call[0][0] for call in mock_workflow.add_node.call_args_list]
        assert "router" in node_calls
        assert "weather_tool" in node_calls
        assert "generate_response" in node_calls
        
        # Conditional edge from router includes weather_tool
        conditional_call = mock_workflow.add_conditional_edges.call_args
        assert conditional_call[0][0] == "router"
        assert "weather_tool" in conditional_call[0][2]
    
    @patch('src.nodes.graph.StateGraph')
    def test_graph_has_correct_node_flow_rag(self, mock_state_graph):
        """Test graph structure for RAG flow"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Verify the flow: router -> (conditional) -> rag_tool -> generate_response -> END
        node_calls = [call[0][0] for call in mock_workflow.add_node.call_args_list]
        assert "router" in node_calls
        assert "rag_tool" in node_calls
        assert "generate_response" in node_calls
        
        # Conditional edge from router includes rag_tool
        conditional_call = mock_workflow.add_conditional_edges.call_args
        assert conditional_call[0][0] == "router"
        assert "rag_tool" in conditional_call[0][2]
    
    @patch('src.nodes.graph.StateGraph')
    def test_graph_conditional_routing_mapping(self, mock_state_graph):
        """Test that conditional routing has correct mapping"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Get the routing mapping
        conditional_call = mock_workflow.add_conditional_edges.call_args
        routing_map = conditional_call[0][2]
        
        assert routing_map["weather_tool"] == "weather_tool"
        assert routing_map["rag_tool"] == "rag_tool"
        assert len(routing_map) == 2
    
    @patch('src.nodes.graph.StateGraph')
    def test_graph_all_tools_connect_to_generate_response(self, mock_state_graph):
        """Test that both tools connect to generate_response"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        create_graph()
        
        # Get all edge calls
        edge_calls = mock_workflow.add_edge.call_args_list
        edge_pairs = [(call[0][0], call[0][1]) for call in edge_calls]
        
        assert ("weather_tool", "generate_response") in edge_pairs
        assert ("rag_tool", "generate_response") in edge_pairs


class TestGraphExecution:
    """Test suite for graph execution behavior"""
    
    @patch('src.nodes.graph.StateGraph')
    @patch('src.nodes.graph.router_node')
    @patch('src.nodes.graph.weather_tool_node')
    @patch('src.nodes.graph.generate_response_node')
    def test_create_graph_can_be_invoked(
        self, mock_gen_node, mock_weather_node, mock_router_node, mock_state_graph
    ):
        """Test that created graph can be invoked"""
        mock_workflow = Mock()
        mock_compiled_app = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_compiled_app
        
        app = create_graph()
        
        # Verify the app has invoke method (it's a compiled graph)
        assert app == mock_compiled_app
        assert hasattr(app, 'invoke') or app == mock_compiled_app
    
    @patch('src.nodes.graph.StateGraph')
    def test_create_graph_multiple_calls_create_separate_graphs(self, mock_state_graph):
        """Test that multiple calls to create_graph create separate instances"""
        mock_workflow1 = Mock()
        mock_workflow2 = Mock()
        mock_app1 = Mock()
        mock_app2 = Mock()
        
        mock_state_graph.side_effect = [mock_workflow1, mock_workflow2]
        mock_workflow1.compile.return_value = mock_app1
        mock_workflow2.compile.return_value = mock_app2
        
        app1 = create_graph()
        app2 = create_graph()
        
        assert app1 != app2
        assert mock_state_graph.call_count == 2


class TestGraphNodeOrdering:
    """Test suite for node addition ordering"""
    
    @patch('src.nodes.graph.StateGraph')
    def test_nodes_added_before_edges(self, mock_state_graph):
        """Test that nodes are added before edges"""
        mock_workflow = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = Mock()
        
        call_order = []
        
        def track_add_node(*args, **kwargs):
            call_order.append('add_node')
        
        def track_add_edge(*args, **kwargs):
            call_order.append('add_edge')
        
        def track_add_conditional_edges(*args, **kwargs):
            call_order.append('add_conditional_edges')
        
        def track_set_entry_point(*args, **kwargs):
            call_order.append('set_entry_point')
        
        mock_workflow.add_node.side_effect = track_add_node
        mock_workflow.add_edge.side_effect = track_add_edge
        mock_workflow.add_conditional_edges.side_effect = track_add_conditional_edges
        mock_workflow.set_entry_point.side_effect = track_set_entry_point
        
        create_graph()
        
        # All add_node calls should come before edge operations
        first_edge_index = call_order.index('add_edge') if 'add_edge' in call_order else len(call_order)
        first_node_index = call_order.index('add_node')
        
        # Count how many add_node calls come before first edge
        nodes_before_edges = sum(1 for call in call_order[:first_edge_index] if call == 'add_node')
        
        # Should have all 4 nodes added before edges
        assert nodes_before_edges == 4


# Fixtures for common test setup
@pytest.fixture
def mock_compiled_graph():
    """Fixture providing a mock compiled graph"""
    graph = Mock()
    graph.invoke.return_value = {
        "query": "test",
        "intent": "weather",
        "final_response": "Test response",
        "pdf_context": "",
        "weather_data": {}
    }
    return graph


@pytest.fixture
def sample_graph_input():
    """Fixture providing sample input for graph execution"""
    return AgentState(
        query="What's the weather in London?",
        intent="",
        pdf_context="",
        final_response="",
        weather_data=None
    )