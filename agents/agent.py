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


# Define vendor prompt templates
vendor_system_template = PromptTemplate(
    input_variables=["brand", "plan", "sales", "cash_flow", "products"],
    template="""You are a vendor agent in a competitive ONU device market.
Your current state:
- Brand: {brand}
- Strategy: {plan}
- Current sales: {sales}
- Current cash flow: {cash_flow}
- Products: {products}

Make decisions to maximize your market share and profitability."""
)

vendor_decision_template = PromptTemplate(
    input_variables=["current_state"],
    template="""Based on your current state and the following market conditions:
{current_state}

Make decisions about:
1. Product pricing (adjust current prices by a percentage)
2. Marketing investments (allocate budget)
3. R&D investments (allocate budget)
4. Production levels (set target)

Your response MUST be a valid JSON object with these exact keys:
{{
    "price_adjustment": <float between -0.2 and 0.2>,
    "marketing_budget": <integer between 10000 and 50000>,
    "rd_budget": <integer between 10000 and 50000>,
    "production_target": <integer between 100 and 1000>
}}"""
)

# Define ISP prompt templates
isp_system_template = PromptTemplate(
    input_variables=["cash_flow", "buy", "min_quality_score"],
    template="""You are an ISP agent purchasing ONU devices.
Your current state:
- Budget: {cash_flow}
- Current purchases: {buy}
- Quality requirements: {min_quality_score}

Make decisions to optimize your purchases based on quality and cost."""
)

isp_decision_template = PromptTemplate(
    input_variables=["current_state"],
    template="""Based on your current state and the following vendor evaluations:
{current_state}

Make decisions about:
1. Which vendor to purchase from (vendor1 or vendor2)
2. How many units to purchase (based on your budget)
3. Quality requirements (minimum acceptable score)

Your response MUST be a valid JSON object with these exact keys:
{{
    "selected_vendor": "vendor1" or "vendor2",
    "purchase_quantity": <integer between 1 and 100>,
    "min_quality_score": <float between 0 and 1>
}}"""
)

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

class VendorAgent(BaseAgent):
    def __init__(self, agent_id: str, initial_state: Dict[str, Any]):
        super().__init__(agent_id, initial_state)
        self.strategy = None
        self.market_tool = MarketAnalysisTool()
        self.strategy_tool = StrategyPlanningTool()
        
    def _analyze_and_plan(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run market_analysis and strategy_planning once and log usage"""
        market_data = {
            'products': self.state.current_state['products'],
            'prices': [p['price'] for p in self.state.current_state['products']],
            'market_share': self.state.current_state.get('sales', 0),
            'cash_flow': self.state.current_state['cash_flow']
        }
        market_analysis = self.market_tool._run(market_data)
        LoggingUtils.log_tool_usage(
            self.tool_usage_log, self.agent_id, 'market_analysis', market_data, market_analysis
        )
        strategy_data = {
            'current_state': self.state.current_state,
            'market_analysis': market_analysis,
            'plan': self.state.current_state['plan']
        }
        strategy_plan = self.strategy_tool._run(strategy_data)
        LoggingUtils.log_tool_usage(
            self.tool_usage_log, self.agent_id, 'strategy_planning', strategy_data, strategy_plan
        )
        return market_analysis, strategy_plan

    def _get_system_prompt(self) -> str:
        return vendor_system_template.format(
            brand=self.state.current_state['brand'],
            plan=self.state.current_state['plan'],
            sales=self.state.current_state['sales'],
            cash_flow=self.state.current_state['cash_flow'],
            products=json.dumps(self.state.current_state['products'], indent=2)
        )
        
    def _get_decision_prompt(self) -> str:
        # reuse precomputed analysis and strategy
        enhanced_state = {
            **self.state.current_state,
            'market_analysis': self.market_analysis,
            'strategy_plan': self.strategy_plan
        }
        return vendor_decision_template.format(
            current_state=json.dumps(enhanced_state, indent=2)
        )
        
    def make_decision(self) -> Dict[str, Any]:
        """Compute analysis once, set required keys, and call common make_decision"""
        self._required_keys = ['price_adjustment', 'marketing_budget', 'rd_budget', 'production_target']
        # perform market analysis and strategy planning once
        self.market_analysis, self.strategy_plan = self._analyze_and_plan()
        return super().make_decision()

class ISPAgent(BaseAgent):
    def __init__(self, agent_id: str, initial_state: Dict[str, Any]):
        super().__init__(agent_id, initial_state)
        self.portfolio_tool = PortfolioAnalysisTool()
        
    def _get_system_prompt(self) -> str:
        return isp_system_template.format(
            cash_flow=self.state.current_state['cash_flow'],
            buy=self.state.current_state['buy'],
            min_quality_score=self.state.current_state.get('min_quality_score', 0)
        )
        
    def _get_decision_prompt(self) -> str:
        # Include vendor evaluations to provide meaningful comparison data
        vendor_evaluations = self.state.current_state.get('vendor_evaluations', {})
        
        # Create an enhanced state object with all the information needed for decision-making
        enhanced_state = {
            **self.state.current_state,
            'vendor_details': vendor_evaluations
        }
        
        return isp_decision_template.format(
            current_state=json.dumps(enhanced_state, indent=2)
        )
        
    def make_decision(self, vendors: List[Any]) -> Dict[str, Any]:
        """Evaluate vendors internally and then make a purchase decision"""
        self._required_keys = ['selected_vendor', 'purchase_quantity', 'min_quality_score']
        # run portfolio analysis on provided vendors
        evaluations = self.evaluate_vendors(vendors)
        # store evaluations into current state for decision prompt
        self.state.current_state['vendor_evaluations'] = evaluations
        return super().make_decision()

    def evaluate_vendors(self, vendors: List[Any]) -> Dict[str, float]:
        """Evaluate available vendors based on defined criteria"""
        # Prepare batch input for portfolio analysis
        batch_input = {
            'vendors': [
                {
                    'vendor_id': v.agent_id,
                    'products': v.state.current_state['products'],
                    'marketing': v.state.current_state.get('content', ''),
                    'rd_budget': v.state.current_state.get('salary_rd', 0)
                }
                for v in vendors
            ]
        }
        # Single batch tool call
        batch_results = self.portfolio_tool._run(batch_input)
        LoggingUtils.log_tool_usage(
            self.tool_usage_log,
            self.agent_id,
            'portfolio_analysis',
            batch_input,
            batch_results
        )
        # Compute evaluations from batch results
        evaluations = {}
        for v in vendors:
            pa = batch_results.get(v.agent_id, {})
            portfolio_score = pa.get('score', 0.5)
            quality_score = (
                v.state.current_state.get('score_product', 0.5) +
                v.state.current_state.get('score_add', 0.5)
            ) / 2
            evaluations[v.agent_id] = {
                'portfolio_score': portfolio_score,
                'quality_score': quality_score,
                'total_score': (portfolio_score + quality_score) / 2
            }
        return evaluations