from agents.agent import BaseAgent
from langchain.prompts import PromptTemplate
from agents.tools import MarketAnalysisTool, StrategyPlanningTool
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from agents.utils import LoggingUtils
import json

# Define vendor prompt templates
vendor_system_template = PromptTemplate(
    input_variables=["description", "brand", "plan", "sales", "cash_flow", "products"],
    template="""You are a vendor agent in a competitive ONU device market.
Your description: {description}
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

Your response MUST be a valid JSON object with these exact keys:
{{
    "price_adjustment": <float between -0.2 and 0.2>,
    "marketing_budget": <integer between 10000 and 50000>,
    "rd_budget": <integer between 10000 and 50000>
}}"""
)

class VendorAgent(BaseAgent):
    def __init__(self, agent_id: str, initial_state: Dict[str, Any], rd_percentage: float = 0.10, capital_percentage: float = 0.02, marketing_percentage: float = 0.08, marketing_capital_percentage: float = 0.01):
        super().__init__(agent_id, initial_state)
        self.strategy = None
        self.market_tool = MarketAnalysisTool()
        self.strategy_tool = StrategyPlanningTool()
        self.name = agent_id  # <-- add this line
        # Configurable percentages for budget calculations
        self.rd_percentage = rd_percentage
        self.capital_percentage = capital_percentage
        self.marketing_percentage = marketing_percentage
        self.marketing_capital_percentage = marketing_capital_percentage

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

        # Update rival_info based on market analysis
        self.state.current_state['rival_info'] = market_analysis.get('rival_info', {})

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
            description=self.state.current_state.get('description', ''),
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
        """Compute analysis once, set required keys, and call common make_decision. R&D and marketing budgets depend on cash and capital, with a higher clamp for dynamic behavior."""
        self._required_keys = ['price_adjustment', 'marketing_budget', 'rd_budget']
        self.market_analysis, self.strategy_plan = self._analyze_and_plan()
        cash = self.state.current_state.get('cash_flow', 0)
        capital = self.state.current_state.get('capital', 0)
        # Use configurable percentages for budget calculations
        rd_budget = int(self.rd_percentage * cash + self.capital_percentage * capital)
        marketing_budget = int(self.marketing_percentage * cash + self.marketing_capital_percentage * capital)
        # Clamp to a higher max for large companies
        rd_budget = max(10000, min(rd_budget, 200000))
        marketing_budget = max(10000, min(marketing_budget, 200000))
        decision = super().make_decision()
        decision['rd_budget'] = rd_budget
        decision['marketing_budget'] = marketing_budget
        return decision