from typing import Dict, List, Any
import numpy as np

def analyze_market_share(vendors: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate market share for each vendor"""
    total_sales = sum(v['sales'] for v in vendors)
    return {v['id']: v['sales']/total_sales for v in vendors}

def calculate_market_growth(historical_data: List[Dict[str, Any]]) -> float:
    """Calculate market growth rate"""
    if len(historical_data) < 2:
        return 0.0
    current = historical_data[-1]['market_size']
    previous = historical_data[-2]['market_size']
    return (current - previous) / previous

def analyze_competition(vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze competitive landscape"""
    return {
        'market_concentration': calculate_herfindahl_index(vendors),
        'price_competition': analyze_price_competition(vendors),
        'product_differentiation': analyze_product_differentiation(vendors)
    }

def calculate_herfindahl_index(vendors: List[Dict[str, Any]]) -> float:
    """Calculate Herfindahl-Hirschman Index for market concentration"""
    market_shares = [v['market_share'] for v in vendors]
    return sum(share ** 2 for share in market_shares)

def analyze_price_competition(vendors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze price competition among vendors"""
    prices = [v['average_price'] for v in vendors]
    return {
        'price_range': max(prices) - min(prices),
        'price_variance': np.var(prices),
        'average_price': np.mean(prices)
    }

def analyze_product_differentiation(vendors: List[Dict[str, Any]]) -> float:
    """Calculate product differentiation index"""
    # TODO: Implement product differentiation analysis
    return 0.0 