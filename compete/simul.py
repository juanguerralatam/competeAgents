from typing import Dict, List, Any
from agents.vendor import VendorAgent
from agents.buyer import BuyerAgent
import json
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class Simulation:
    def __init__(self, market, duration: int = 12, exp_name: str = "default"):
        self.market = market
        self.vendors: List[VendorAgent] = []
        self.buyers: List[BuyerAgent] = []
        self.current_step = 0
        self.duration = duration
        self.start_time = datetime.now()
        self.exp_name = exp_name
        self.buyer_decisions = []  # Store buyer decisions
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize CSV file for buyer decisions
        self.buyer_decisions_file = f'logs/buyer_decisions_{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(self.buyer_decisions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['month', 'buyer_id', 'selected_vendor', 'purchase_quantity', 'min_quality_score'])
            
        # Initialize CSV file for vendor decisions
        self.vendor_decisions_file = f'logs/vendor_decisions_{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(self.vendor_decisions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['month', 'vendor_id', 'rd_investment', 'marketing_investment', 'onu_price', 'quantity_sold'])
        
        logging.info(f"Simulation initialized with duration of {duration} months")
        
    def initialize_agents(self):
        """Initialize vendor and buyer agents from profiles.json"""
        logging.info("Initializing agents from profiles.json...")
        profiles_path = os.path.join(os.path.dirname(__file__), 'profiles.json')
        with open(profiles_path, 'r') as f:
            profiles = json.load(f)
        self.vendors = [VendorAgent(v['agent_id'], v['state']) for v in profiles.get('vendors', [])]
        self.buyers = [BuyerAgent(i['agent_id'], i['state']) for i in profiles.get('isps', [])]
        logging.info(f"Initialized {len(self.vendors)} vendors and {len(self.buyers)} buyers from profiles.json")
        
    def _log_buyer_decision(self, buyer_id: str, decision: Dict[str, Any]):
        """Log buyer decision to CSV file"""
        with open(self.buyer_decisions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_step,
                buyer_id,
                decision['selected_vendor'],
                decision['purchase_quantity'],
                decision['min_quality_score']
            ])
        self.buyer_decisions.append({
            'month': self.current_step,
            'buyer_id': buyer_id,
            'decision': decision
        })

    def _log_vendor_decision(self, vendor: VendorAgent):
        """Log vendor decision to CSV file: month, vendor_id, rd_investment, marketing_investment, onu_price (int), quantity_sold (aggregate from buyer decisions)"""
        # Aggregate quantity sold for this vendor in this month
        quantity_sold = sum(
            d['decision']['purchase_quantity']
            for d in self.buyer_decisions
            if d['decision']['selected_vendor'] == vendor.agent_id and d['month'] == self.current_step
        )
        rd_investment = vendor.state.current_state.get('salary_rd', 0)
        marketing_investment = vendor.state.current_state.get('salary_maketing', 0)
        onu_price = int(vendor.state.current_state['products'][0]['price']) if vendor.state.current_state.get('products') else 0
        with open(self.vendor_decisions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_step,
                vendor.agent_id,
                rd_investment,
                marketing_investment,
                onu_price,
                quantity_sold
            ])
        
    def _update_buyer_finances(self):
        """Update buyer finances based on market growth and installed ONUs."""
        economy_grown = self.market.market_state['market_growth']['economy_grown']
        monthly_revenue_per_onu = 20  # You can adjust this value as needed
        for buyer in self.buyers:
            country_dev = buyer.state.current_state.get('country_development', 1)
            if 'installed_onus' not in buyer.state.current_state:
                buyer.state.current_state['installed_onus'] = 0
            buyer.state.current_state['installed_onus'] += buyer.state.current_state.get('buy', 0)
            buyer.state.current_state['cash_flow'] += buyer.state.current_state['installed_onus'] * monthly_revenue_per_onu
            buyer.state.current_state['cash_flow'] *= (1 + economy_grown * country_dev)

    def _execute_vendor_turns(self):
        """Get vendor decisions and apply them."""
        for vendor in self.vendors:
            logging.debug(f"Getting decision from vendor {vendor.agent_id}")
            decision = vendor.make_decision()
            self._apply_vendor_decision(vendor, decision)
            logging.info(f"Vendor {vendor.agent_id} made decision: {json.dumps(decision, indent=2)}")

    def _execute_buyer_turns(self):
        """Get buyer decisions and apply them."""
        for buyer in self.buyers:
            logging.debug(f"Getting decision from buyer {buyer.agent_id}")
            decision = buyer.make_decision(self.vendors)
            self._apply_buyer_decision(buyer, decision)

    def _update_market_metrics(self):
        """Update buyer cash flow and log vendor decisions."""
        for buyer in self.buyers:
            buy = buyer.state.current_state.get('buy', 0)
            last_decision = next((d['decision'] for d in reversed(self.buyer_decisions) if d['buyer_id'] == buyer.agent_id and d['month'] == self.current_step), None)
            if last_decision:
                vendor_id = last_decision['selected_vendor']
                vendor = next((v for v in self.vendors if v.agent_id == vendor_id), None)
                if vendor:
                    vendor_products = vendor.state.current_state.get('products', [])
                    if vendor_products:
                        onu_cost_price = vendor_products[0].get('price', 1)
                        onu_sold_price = onu_cost_price * 1.2
                        profit = (onu_sold_price - onu_cost_price) * buy
                        buyer.state.current_state['cash_flow'] += profit
        for vendor in self.vendors:
            self._log_vendor_decision(vendor)

    def run_step(self):
        """Run a single simulation step."""
        logging.info(f"Running step {self.current_step + 1}/{self.duration}")
        
        # Update market state
        self.market.update_market_state()
        logging.debug("Market state updated")
        
        self._update_buyer_finances()
        self._execute_vendor_turns()
        self._execute_buyer_turns()
        self._update_market_metrics()

    def _apply_vendor_decision(self, vendor: VendorAgent, decision: Dict[str, Any]):
        """Apply vendor's LLM decision to their state and update cash flow dynamically."""
        logging.debug(f"Applying vendor {vendor.agent_id} decision")
        # Update product prices
        for product in vendor.state.current_state['products']:
            product['price'] *= (1 + decision.get('price_adjustment', 0))
        # Update investments
        vendor.state.current_state['salary_rd'] = decision.get('rd_budget', vendor.state.current_state['salary_rd'])
        vendor.state.current_state['salary_maketing'] = decision.get('marketing_budget', vendor.state.current_state['salary_maketing'])
        # Dynamically update cash_flow: add profit from sales, subtract R&D and marketing investments
        sales = vendor.state.current_state.get('sales', 0)
        price = vendor.state.current_state['products'][0]['price'] if vendor.state.current_state.get('products') else 0
        cost = vendor.state.current_state['products'][0]['cost'] if vendor.state.current_state.get('products') else 0
        profit = (price - cost) * sales
        vendor.state.current_state['cash_flow'] += profit
        vendor.state.current_state['cash_flow'] -= vendor.state.current_state['salary_rd']
        vendor.state.current_state['cash_flow'] -= vendor.state.current_state['salary_maketing']
        # Optionally, reset sales for next period
        vendor.state.current_state['sales'] = 0

    def _apply_buyer_decision(self, buyer: BuyerAgent, decision: Dict[str, Any]):
        """Apply buyer decision and log it, making purchase dynamic based on budget and market state"""
        economy_grown = self.market.market_state.get('market_growth', {}).get('economy_grown', 0.1)
        budget = buyer.state.current_state.get('cash_flow', 0)
        min_quality_score = decision.get('min_quality_score', 0.5)
        selected_vendor = next(v for v in self.vendors if v.agent_id == decision['selected_vendor'])
        vendor_products = selected_vendor.state.current_state.get('products', [])
        if vendor_products:
            product_price = vendor_products[0].get('price', 1)
        else:
            product_price = 1
        max_affordable = int((budget / product_price) * (1 + economy_grown))
        purchase_quantity = max(1, max_affordable)
        # Subtract purchase cost from buyer's cash_flow
        buyer.state.current_state['cash_flow'] -= purchase_quantity * product_price
        # Log and store decision
        self._log_buyer_decision(buyer.agent_id, {
            'selected_vendor': selected_vendor.agent_id,
            'purchase_quantity': purchase_quantity,
            'min_quality_score': min_quality_score
        })
        buyer.state.current_state['buy'] = purchase_quantity
        buyer.state.current_state['min_quality_score'] = min_quality_score
        selected_vendor.state.current_state['sales'] += purchase_quantity
        logging.info(f"Buyer {buyer.agent_id} made decision: {{'selected_vendor': '{selected_vendor.agent_id}', 'purchase_quantity': {purchase_quantity}, 'min_quality_score': {min_quality_score}}}")
        
    def run(self, debug=False):
        """Run the complete simulation"""
        if debug:
            logging.setLevel(logging.DEBUG)
        logging.info("Starting simulation")
        self.initialize_agents()
        
        try:
            for step in tqdm(range(self.duration)):
                self.run_step()
                self.current_step += 1
                logging.info(f"Completed step {self.current_step}/{self.duration}")
                
            # Calculate and log simulation duration
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            logging.info(f"Simulation completed in {duration:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Simulation failed at step {self.current_step}: {str(e)}")
            raise
            
    def is_simulation_complete(self) -> bool:
        """Check if simulation should end"""
        return self.current_step >= self.duration