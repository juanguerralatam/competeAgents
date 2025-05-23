from typing import Dict, List, Any
from langchain.tools import BaseTool

class MarketAnalysisTool(BaseTool):
    name: str = "market_analysis"
    description: str = "Analyze current market conditions and trends"
    
    def _run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return insights"""
        products = market_data.get('products', [])
        prices = market_data.get('prices', [])
        
        # Basic analysis
        avg_price = sum(prices) / len(prices) if prices else 0
        avg_cost = sum([p.get('cost', 0) for p in products]) / len(products) if products else 0
        profit_margin = (avg_price - avg_cost) / avg_price if avg_price > 0 else 0
        
        return {
            'average_price': avg_price,
            'profit_margin': profit_margin}
        
    async def _arun(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(market_data)
    
class StrategyPlanningTool(BaseTool):
    name: str = "strategy_planning"
    description: str = "Plan and update business strategies"
    
    def _run(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan and update business strategies"""
        current_state = strategy_data.get('current_state', {})
        market_analysis = strategy_data.get('market_analysis', {})
        plan = strategy_data.get('plan', 'cost_leadership')
        
        # Basic strategy recommendations based on plan type
        if plan == 'cost_leadership':
            return {
                'focus': 'operational_efficiency',
                'pricing_strategy': 'competitive_pricing',
                'market_approach': 'mass_market',
                'production_recommendation': 'scale_up',
                'next_steps': [
                    'Optimize production costs',
                    'Increase production volume',
                    'Expand market reach'
                ]
            }
        else:  # cost_focus
            return {
                'focus': 'product_differentiation',
                'pricing_strategy': 'value_based_pricing',
                'market_approach': 'niche_market',
                'production_recommendation': 'quality_over_quantity',
                'next_steps': [
                    'Enhance product features',
                    'Target specific customer segments',
                    'Invest in brand development'
                ]
            }
        
    async def _arun(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(strategy_data)
    

class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze vendor product portfolios"
    
    def _run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product portfolios and return scores"""
        # Batch mode: multiple vendors
        if 'vendors' in portfolio_data:
            results = {}
            for entry in portfolio_data['vendors']:
                vid = entry.get('vendor_id')
                pd = {
                    'products': entry.get('products', []),
                    'marketing': entry.get('marketing', ''),
                    'rd_budget': entry.get('rd_budget', 0)
                }
                # compute single vendor analysis by calling single-vendor branch
                single = self._run(pd)
                results[vid] = single
            return results

        # Single vendor mode
        products = portfolio_data.get('products', [])
        marketing = portfolio_data.get('marketing', '')
        rd_budget = portfolio_data.get('rd_budget', 0)
        # Basic portfolio evaluation metrics
        avg_price = sum([p.get('price', 0) for p in products]) / len(products) if products else 0
        avg_cost = sum([p.get('cost', 0) for p in products]) / len(products) if products else 0
        margin = (avg_price - avg_cost) / avg_price if avg_price > 0 else 0
        
        # Marketing evaluation - simple keyword based
        marketing_score = 0.5  # default
        if 'Premium' in marketing:
            marketing_score = 0.7
        elif 'Advanced' in marketing:
            marketing_score = 0.6
        
        # R&D evaluation
        rd_score = min(1.0, rd_budget / 50000)  # Normalize to 0-1 range
        
        # Calculate composite score
        product_score = margin * 0.6 + (1.0 if len(products) > 1 else 0.5) * 0.4
        composite_score = product_score * 0.5 + marketing_score * 0.3 + rd_score * 0.2
        
        return {
            'score': round(composite_score, 2),
            }
        
    async def _arun(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(portfolio_data)