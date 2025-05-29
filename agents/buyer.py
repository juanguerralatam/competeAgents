from langchain.prompts import PromptTemplate
from agents.tools import PortfolioAnalysisTool
from typing import Dict, List, Any
from agents.agent import BaseAgent
from langchain.prompts import PromptTemplate
import json

isp_system_template = PromptTemplate(
    input_variables=["cash_flow", "buy", "min_quality_score", "country_development"],
    template="""You are a buyer agent purchasing ONU devices.
Your current state:
- Budget: {cash_flow}
- Current purchases: {buy}
- Quality requirements: {min_quality_score}
- Country development: {country_development}

Make decisions to optimize your purchases based on quality, cost, and your country's development level."""
)

isp_decision_template = PromptTemplate(
    input_variables=["current_state"],
    template="""Based on your current state and the following vendor evaluations:
{current_state}

Make decisions about:
1. Which vendor to purchase from (vendor1 or vendor2)
2. How many units to purchase (based on your budget and number of customers)
3. Quality requirements (minimum acceptable score)

Your response MUST be a valid JSON object with these exact keys:
{{
    "selected_vendor": "vendor1" or "vendor2",
    "purchase_quantity": <integer>,
    "min_quality_score": <float between 0 and 1>
}}"""
)

class BuyerAgent(BaseAgent):
    def __init__(self, agent_id: str, initial_state: Dict[str, Any]):
        super().__init__(agent_id, initial_state)
        self.portfolio_tool = PortfolioAnalysisTool()
        
    def _get_system_prompt(self) -> str:
        return isp_system_template.format(
            cash_flow=self.state.current_state['cash_flow'],
            buy=self.state.current_state['buy'],
            min_quality_score=self.state.current_state.get('min_quality_score', 0),
            country_development=self.state.current_state.get('country_development', 1)
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
        
    def _select_best_vendor(self, vendors: List[Any], min_quality: float, strategy: str) -> Any:
        """Select the best vendor based on the buyer's strategy and country development."""
        best_score = float('-inf')
        best_vendor = None
        country_dev = self.state.current_state.get('country_development', 1)

        for v in vendors:
            products = v.state.current_state.get('products', [])
            quality = v.state.current_state.get('score_product', 0)
            if products:
                price = products[0].get('price', 1)

                if strategy == 'cost_sensitive':
                    # Weighted score for cost-sensitive buyers
                    price_weight = 0.7 if country_dev < 0.5 else 0.5
                    quality_weight = 0.3 if country_dev < 1 else 0.5
                    score = (quality * quality_weight) - (price * price_weight)
                elif strategy == 'premium':
                    # Premium buyers prioritize quality
                    score = quality
                else:  # Default strategy
                    # Default scoring logic
                    score = quality - price

                if quality >= min_quality and score > best_score:
                    best_score = score
                    best_vendor = v

        return best_vendor

    def make_decision(self, vendors: List[Any]) -> Dict[str, Any]:
        """Evaluate vendors and make a purchase decision based on ISP's customer focus."""
        description = self.state.current_state.get('description', '').lower()
        min_quality = 0.5

        if 'cost sensitive' in description:
            strategy = 'cost_sensitive'
        elif 'premium' in description:
            strategy = 'premium'
        else:
            strategy = 'default'

        best_vendor = self._select_best_vendor(vendors, min_quality, strategy)

        if best_vendor is None:
            return {
                'selected_vendor': None,
                'min_quality_score': min_quality
            }

        return {
            'selected_vendor': best_vendor.agent_id,
            'min_quality_score': best_vendor.state.current_state.get('score_product', min_quality)
        }