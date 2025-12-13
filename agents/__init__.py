"""Multi-agent investment analysis system."""

from .base_agent import BaseAgent
from .summary_agent import SummaryAgent
from .swot_agent import SWOTAgent
from .metrics_agent import MetricsAgent
from .decision_agent import DecisionAgent

# V2 agents with enhanced features
from .summary_agent_v2 import SummaryAgentV2
from .swot_agent_v2 import SWOTAgentV2
from .metrics_agent_v2 import MetricsAgentV2
from .decision_agent_v2 import DecisionAgentV2

__all__ = [
    'BaseAgent',
    'SummaryAgent',
    'SWOTAgent',
    'MetricsAgent',
    'DecisionAgent',
    'SummaryAgentV2',
    'SWOTAgentV2',
    'MetricsAgentV2',
    'DecisionAgentV2'
]
