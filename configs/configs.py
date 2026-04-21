"""
Shared dataclasses for the multi-agent debate system.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentResponse:
    """A single agent's output for one debate round."""
    name: str
    reasoning: str        # private chain-of-thought (not shared with peers)
    answer: str           # the extracted answer letter (e.g., "A", "B")
    public_message: str = ""  # concise debate message shared with peers (ThinkThenSpeakDebater only)


@dataclass
class RoundEntry:
    """One round of the debate, containing every agent's response."""
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    # key = agent_id (e.g., "agent_0"), value = that agent's response


# A full conversation is simply: List[RoundEntry]
# Index 0 = round 0, index 1 = round 1, etc.
Conversation = List[RoundEntry]
