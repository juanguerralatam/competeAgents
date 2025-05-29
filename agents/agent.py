from typing import Dict, List, Any
from dataclasses import dataclass
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import json
import logging
from datetime import datetime
from agents.tools import PortfolioAnalysisTool, MarketAnalysisTool, StrategyPlanningTool
from agents.utils import LoggingUtils

@dataclass
class AgentState:
    memory: List[Dict[str, Any]]
    current_state: Dict[str, Any]
    
class BaseAgent:
    def __init__(self, agent_id: str, initial_state: Dict[str, Any]):
        self.agent_id = agent_id
        self.state = AgentState(
            memory=[],
            current_state=initial_state
        )
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1000,
            max_retries=10,
        )
        
        # Set up logging
        LoggingUtils.setup_logs_directory()
        self.conversation_log, self.tool_usage_log = LoggingUtils.get_log_filenames(self.agent_id)
        
    def update_memory(self, new_state: Dict[str, Any]):
        """Update agent's memory with new state"""
        self.state.memory.append(new_state)
        if len(self.state.memory) > 3:
            self.state.memory.pop(0)
            
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        raise NotImplementedError
        
    def _get_decision_prompt(self) -> str:
        """Get the decision-making prompt for the agent"""
        raise NotImplementedError
        
    def make_decision(self) -> Dict[str, Any]:
        """Make a decision using LLM based on current state and memory"""
        system_prompt = self._get_system_prompt()
        decision_prompt = self._get_decision_prompt()
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=decision_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Log the conversation
        LoggingUtils.log_conversation(
            self.conversation_log,
            self.agent_id,
            system_prompt,
            decision_prompt,
            response.content
        )
        
        return self._parse_llm_response(response.content, self._required_keys)

    def _parse_llm_response(self, response: str, required_keys: list) -> Dict[str, Any]:
        """Unified JSON extraction and validation"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end <= 0:
                logging.error(f"Could not find JSON in response: {response}")
                return {}
            json_str = response[start:end]
            decision = json.loads(json_str)
            missing = [k for k in required_keys if k not in decision]
            if missing:
                logging.error(f"Missing required keys {missing} in decision: {decision}")
                return {}
            return decision
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return {}
