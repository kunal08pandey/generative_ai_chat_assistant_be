"""
Document-to-Report Bounded Agent
================================

Implements an end-to-end, bounded agent workflow (Phase 6):
  1. Retrieve & inspect document excerpts scoped strictly to owner.
  2. Extract key metrics, constraints, and findings.
  3. Synthesize an executive report with structured recommendations.

Enforces a typed tool allowlist to prevent unrestricted execution.
Emits intermediate steps as collapsible reasoning blocks (<think>...</think>).
"""
from __future__ import annotations

import json
from enum import Enum
from typing import AsyncGenerator, Dict, Any, Optional

from src.llm.base import BaseLLM
from src.rag.retriever import retrieve_context
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentToolName(str, Enum):
    """Allowlist of permitted tools for the Document-to-Report agent."""
    SEARCH_DOCUMENTS = "search_documents"
    EXTRACT_FINDINGS = "extract_findings"
    SYNTHESIZE_EXECUTIVE_SUMMARY = "synthesize_executive_summary"


class DocumentReportAgent:
    """
    Bounded agent for generating executive summary reports from documents.
    """

    ALLOWED_TOOLS = {
        AgentToolName.SEARCH_DOCUMENTS,
        AgentToolName.EXTRACT_FINDINGS,
        AgentToolName.SYNTHESIZE_EXECUTIVE_SUMMARY,
    }

    def __init__(self, llm: BaseLLM, owner_id: str):
        self.llm = llm
        self.owner_id = owner_id

    async def execute_tool(self, tool_name: AgentToolName, params: Dict[str, Any]) -> str:
        """
        Execute a strictly allowlisted tool. Rejects unauthorized tool names.
        """
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(f"Tool '{tool_name}' is not in the agent allowlist.")

        if tool_name == AgentToolName.SEARCH_DOCUMENTS:
            query = params.get("query", "")
            return await retrieve_context(question=query, owner_id=self.owner_id)

        elif tool_name == AgentToolName.EXTRACT_FINDINGS:
            raw_text = params.get("text", "")
            prompt = (
                "You are an analytical assistant. Extract the top 3-5 critical findings, "
                "data points, and constraints from the following text concisely:\n\n"
                f"{raw_text[:3000]}\n\n"
                "Findings:"
            )
            return await self.llm.generate(prompt)

        elif tool_name == AgentToolName.SYNTHESIZE_EXECUTIVE_SUMMARY:
            context = params.get("context", "")
            topic = params.get("topic", "Executive Report")
            prompt = (
                f"Synthesize an executive summary for: {topic}\n"
                f"Based on evidence:\n{context[:3000]}\n"
                "Executive Summary:"
            )
            return await self.llm.generate(prompt)

        return ""

    async def stream_workflow(
        self,
        topic: str,
        document_query: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Runs the bounded agent workflow:
        Emits <think> reasoning steps first, then streams the structured executive report.
        """
        # Step 1: Tool search_documents
        yield "<think>\n"
        yield "• Step 1/3: Retrieving document evidence from workspace...\n"
        query = document_query or topic
        doc_context = await self.execute_tool(
            AgentToolName.SEARCH_DOCUMENTS,
            {"query": query},
        )

        has_context = bool(doc_context and doc_context.strip())
        if has_context:
            yield f"✓ Retrieved context ({len(doc_context.splitlines())} snippets found).\n\n"
        else:
            yield "ℹ No specific indexed documents found; proceeding with general synthesis.\n\n"

        # Step 2: Tool extract_findings
        yield "• Step 2/3: Analyzing key findings, constraints, and metrics...\n"
        findings = ""
        if has_context:
            findings = await self.execute_tool(
                AgentToolName.EXTRACT_FINDINGS,
                {"text": doc_context},
            )
            yield f"✓ Extracted key findings:\n{findings.strip()}\n\n"
        else:
            yield "✓ Formulated analytical framework.\n\n"

        # Step 3: Synthesis
        yield "• Step 3/3: Synthesizing executive report with actionable recommendations...\n"
        yield "</think>\n\n"

        # Final Report Generation Stream
        report_prompt = (
            "You are a Senior Strategic Advisor. Create an Executive Briefing & Report.\n"
            f"Topic/Objective: {topic}\n"
            f"Key Findings:\n{findings or 'No specific document evidence provided.'}\n"
            f"Raw Excerpts:\n{doc_context[:2000] if has_context else 'None'}\n\n"
            "Format your output cleanly in Markdown with these exact sections:\n"
            "# Executive Report: " + topic + "\n\n"
            "## 1. Executive Summary\n"
            "High-level context, strategic importance, and core conclusions.\n\n"
            "## 2. Key Findings & Data\n"
            "Bullet points with specific findings, numbers, and operational realities.\n\n"
            "## 3. Strategic Recommendations\n"
            "Actionable, prioritized steps (P0, P1, P2) for leadership.\n\n"
            "## 4. Risks & Mitigations\n"
            "Anticipated challenges and contingency measures.\n"
        )

        async for token in self.llm.generate_stream(report_prompt):
            yield token
