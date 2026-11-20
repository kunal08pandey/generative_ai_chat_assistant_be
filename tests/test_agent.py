"""
Document-to-Report Agent Tests
==============================

Tests:
  - Strict enforcement of typed tool allowlist.
  - Rejection of unauthorized tools with PermissionError.
  - End-to-end streaming workflow with intermediate <think> reasoning blocks.
  - Authentication and conversation data isolation on POST /agent/report.
"""
import pytest

from src.agent.document_report_agent import DocumentReportAgent, AgentToolName
from src.llm.base import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self, responses=None):
        self.responses = responses or ["Mocked report output"]

    async def generate(self, prompt: str, **kwargs) -> str:
        return "Key findings extracted from brief."

    async def generate_stream(self, prompt: str, **kwargs):
        yield "# Executive Report: Project Aurora\n\n"
        yield "## 1. Executive Summary\n"
        yield "Strategic analysis completed successfully."


def test_agent_tool_allowlist_enforcement():
    """Verify unauthorized tools are blocked."""
    agent = DocumentReportAgent(llm=MockLLM(), owner_id="user-123")
    assert AgentToolName.SEARCH_DOCUMENTS in agent.ALLOWED_TOOLS
    assert AgentToolName.EXTRACT_FINDINGS in agent.ALLOWED_TOOLS
    assert AgentToolName.SYNTHESIZE_EXECUTIVE_SUMMARY in agent.ALLOWED_TOOLS


@pytest.mark.anyio
async def test_agent_rejects_unauthorized_tool():
    agent = DocumentReportAgent(llm=MockLLM(), owner_id="user-123")
    with pytest.raises(PermissionError):
        await agent.execute_tool("bash_command", {"command": "rm -rf /"})


@pytest.mark.anyio
async def test_agent_workflow_generates_collapsible_reasoning():
    agent = DocumentReportAgent(llm=MockLLM(), owner_id="user-123")
    tokens = []
    async for token in agent.stream_workflow(topic="Q3 Budget Review"):
        tokens.append(token)

    full_output = "".join(tokens)
    assert "<think>" in full_output
    assert "</think>" in full_output
    assert "Step 1/3" in full_output
    assert "# Executive Report:" in full_output


def test_agent_route_unauthenticated_rejected(client):
    res = client.post(
        "/agent/report",
        json={"topic": "Roadmap", "conversation_id": "conv-123"},
    )
    assert res.status_code == 401


def test_agent_route_isolation(auth_client, bob_client, conversation, monkeypatch):
    """User Bob cannot trigger an agent report on Alice's conversation."""
    monkeypatch.setattr("src.api.routes.agent.get_llm_client", lambda *args, **kwargs: MockLLM())

    res = bob_client.post(
        "/agent/report",
        json={"topic": "Roadmap", "conversation_id": conversation["id"]},
    )
    assert res.status_code == 404


def test_agent_route_success(auth_client, conversation, monkeypatch):
    """Authorized user can run agent workflow and receive SSE stream."""
    monkeypatch.setattr("src.api.routes.agent.get_llm_client", lambda *args, **kwargs: MockLLM())

    res = auth_client.post(
        "/agent/report",
        json={"topic": "Quarterly Expansion", "conversation_id": conversation["id"]},
    )
    assert res.status_code == 200
    assert "Executive Report" in res.text
    assert "<think>" in res.text
