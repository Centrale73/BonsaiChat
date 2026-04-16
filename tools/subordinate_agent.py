"""
tools/subordinate_agent.py — Sub-agent delegation tool for BonsaiChat.

Inspired by Agent Zero's call_subordinate / Delegation tool.
Creates a fresh Agno agent with its own session and runs a focused task,
returning the result to the parent agent as a tool response.

The sub-agent uses the same model backend (local llama-server or cloud) as
the parent.  It does NOT share memory or chat history — it starts clean so
the parent's context is not polluted.
"""

from __future__ import annotations

from typing import Optional

from agno.agent import Agent
from agno.tools import Toolkit


_SUBORDINATE_INSTRUCTIONS = (
    "You are a focused sub-agent.  Your only goal is to complete the task "
    "assigned to you as thoroughly and concisely as possible.  "
    "Return only the result — no preamble, no explanation of your process."
)


class SubAgentTools(Toolkit):
    """
    Agno Toolkit that lets the main agent spin up a subordinate agent
    for isolated sub-task completion.

    The parent agent calls `delegate_task(task=...)` and receives the
    sub-agent's final response as a string.
    """

    def __init__(self, parent_model, db=None, knowledge=None):
        super().__init__(name="sub_agent")
        self._model   = parent_model
        self._db      = db
        self._knowledge = knowledge
        self.register(self.delegate_task)

    def delegate_task(self, task: str, specialist_role: Optional[str] = None) -> str:
        """
        Delegate a focused sub-task to a fresh subordinate agent and return
        its response.

        Use this when the task is self-contained and would clutter your own
        context — e.g. 'summarise this document', 'write a specific function',
        'translate this paragraph'.

        Args:
            task:            A clear description of the sub-task to perform.
            specialist_role: Optional persona/role for the sub-agent
                             (e.g. 'You are an expert Python developer').

        Returns:
            The sub-agent's complete response as a string.
        """
        instructions = _SUBORDINATE_INSTRUCTIONS
        if specialist_role:
            instructions = specialist_role + "\n\n" + instructions

        sub_agent = Agent(
            model=self._model,
            db=self._db,
            knowledge=self._knowledge,
            search_knowledge=self._knowledge is not None,
            instructions=instructions,
            markdown=True,
        )

        try:
            response = sub_agent.run(task, stream=False)
            content = response.content if hasattr(response, "content") else str(response)
            return content or "[Sub-agent returned empty response]"
        except Exception as exc:
            return f"[Sub-agent error: {exc}]"
