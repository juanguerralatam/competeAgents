from typing import Dict, Any
import json

class MarketEnvironment:
    def __init__(self):
        self.global_params = self._load_global_params()
        self.market_state = {
            'market_growth': {
                'internet_grown': 0.2,
                'economy_grown': 0.2
            },
            'technological_advancement': {
                'migration_fiber': 0.2
            },
            'environmental_factors': {
                'country': 'H',  # H, M, L
                'company_size': 'L'  # L, S
            }
        }
        
    def _load_global_params(self) -> Dict[str, Any]:
        """Load global parameters from config file"""
        try:
            with open('config/global_params.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_params()
            
    def _get_default_params(self) -> Dict[str, Any]:
        """Return default parameters if config file not found"""
        return {
            'company_basics': {
                'brand': 'Default',
                'fix_cost': 100000,
                'variable_cost': 50000,
                'capital': 1000000,
                'cash_flow': 50000
            },
            'market_growth': {
                'internet_grown': 0.05,
                'economy_grown': 0.03
            },
            'technological_advancement': {
                'migration_fiber': 0.02
            }
        }
        
    def update_market_state(self):
        """Update market state based on global parameters"""
        # Update market growth based on annual parameters
        self.market_state['market_growth']['internet_grown'] = self.global_params['market_growth']['internet_grown']
        self.market_state['market_growth']['economy_grown'] = self.global_params['market_growth']['economy_grown']
        
        # Update technological advancement
        self.market_state['technological_advancement']['migration_fiber'] = self.global_params['technological_advancement']['migration_fiber']
        
        # TODO: Implement market state update logic
        pass