from typing import Dict, List, Any
from agents.agent import VendorAgent, ISPAgent
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
        self.isps: List[ISPAgent] = []
        self.current_step = 0
        self.duration = duration
        self.start_time = datetime.now()
        self.exp_name = exp_name
        self.isp_decisions = []  # Store ISP decisions
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize CSV file for ISP decisions
        self.isp_decisions_file = f'logs/isp_decisions_{exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(self.isp_decisions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['month', 'isp_id', 'selected_vendor', 'purchase_quantity', 'min_quality_score'])
            
        logging.info(f"Simulation initialized with duration of {duration} months")
        
    def initialize_agents(self):
        """Initialize vendor and ISP agents from profiles.json"""
        logging.info("Initializing agents from profiles.json...")
        profiles_path = os.path.join(os.path.dirname(__file__), 'profiles.json')
        with open(profiles_path, 'r') as f:
            profiles = json.load(f)
        self.vendors = [VendorAgent(v['agent_id'], v['state']) for v in profiles.get('vendors', [])]
        self.isps = [ISPAgent(i['agent_id'], i['state']) for i in profiles.get('isps', [])]
        logging.info(f"Initialized {len(self.vendors)} vendors and {len(self.isps)} ISPs from profiles.json")
        
    def _log_isp_decision(self, isp_id: str, decision: Dict[str, Any]):
        """Log ISP decision to CSV file"""
        with open(self.isp_decisions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_step,
                isp_id,
                decision['selected_vendor'],
                decision['purchase_quantity'],
                decision['min_quality_score']
            ])
        self.isp_decisions.append({
            'month': self.current_step,
            'isp_id': isp_id,
            'decision': decision
        })

    def run_step(self):
        """Run a single simulation step"""
        logging.info(f"Running step {self.current_step + 1}/{self.duration}")
        
        # Update market state
        self.market.update_market_state()
        logging.debug("Market state updated")
        
        # Get vendor decisions from LLM
        for vendor in self.vendors:
            logging.debug(f"Getting decision from vendor {vendor.agent_id}")
            decision = vendor.make_decision()
            self._apply_vendor_decision(vendor, decision)
            logging.info(f"Vendor {vendor.agent_id} made decision: {json.dumps(decision, indent=2)}")
            
        # Get ISP decisions from LLM
        for isp in self.isps:
            logging.debug(f"Getting decision from ISP {isp.agent_id}")
            decision = isp.make_decision(self.vendors)
            self._apply_isp_decision(isp, decision)
        
        # After all decisions, update ISP cash_flow based on ONU purchases and profit
        for isp in self.isps:
            buy = isp.state.current_state.get('buy', 0)
            min_quality_score = isp.state.current_state.get('min_quality_score', 0)
            # Find the vendor from the last decision (if any)
            last_decision = next((d['decision'] for d in reversed(self.isp_decisions) if d['isp_id'] == isp.agent_id and d['month'] == self.current_step), None)
            if last_decision:
                vendor_id = last_decision['selected_vendor']
                vendor = next((v for v in self.vendors if v.agent_id == vendor_id), None)
                if vendor:
                    vendor_products = vendor.state.current_state.get('products', [])
                    if vendor_products:
                        onu_cost_price = vendor_products[0].get('price', 1)
                        # Assume ISP sells ONU at 20% markup
                        onu_sold_price = onu_cost_price * 1.2
                        profit = (onu_sold_price - onu_cost_price) * buy
                        isp.state.current_state['cash_flow'] += profit

    def _apply_vendor_decision(self, vendor: VendorAgent, decision: Dict[str, Any]):
        """Apply vendor's LLM decision to their state"""
        logging.debug(f"Applying vendor {vendor.agent_id} decision")
        # Update product prices
        for product in vendor.state.current_state['products']:
            product['price'] *= (1 + decision.get('price_adjustment', 0))
            
        # Update investments
        vendor.state.current_state['salary_rd'] = decision.get('rd_budget', vendor.state.current_state['salary_rd'])
        vendor.state.current_state['salary_maketing'] = decision.get('marketing_budget', vendor.state.current_state['salary_maketing'])
        
    def _apply_isp_decision(self, isp: ISPAgent, decision: Dict[str, Any]):
        """Apply ISP decision and log it, making purchase dynamic based on budget and market state"""
        economy_grown = self.market.market_state.get('market_growth', {}).get('economy_grown', 0.1)
        budget = isp.state.current_state.get('cash_flow', 0)
        min_quality_score = decision.get('min_quality_score', 0.5)
        selected_vendor = next(v for v in self.vendors if v.agent_id == decision['selected_vendor'])
        vendor_products = selected_vendor.state.current_state.get('products', [])
        if vendor_products:
            product_price = vendor_products[0].get('price', 1)
        else:
            product_price = 1
        max_affordable = int((budget / product_price) * (1 + economy_grown))
        purchase_quantity = max(1, max_affordable)
        # Subtract purchase cost from ISP's cash_flow
        isp.state.current_state['cash_flow'] -= purchase_quantity * product_price
        # Log and store decision
        self._log_isp_decision(isp.agent_id, {
            'selected_vendor': selected_vendor.agent_id,
            'purchase_quantity': purchase_quantity,
            'min_quality_score': min_quality_score
        })
        isp.state.current_state['buy'] = purchase_quantity
        isp.state.current_state['min_quality_score'] = min_quality_score
        selected_vendor.state.current_state['sales'] += purchase_quantity
        logging.info(f"ISP {isp.agent_id} made decision: {{'selected_vendor': '{selected_vendor.agent_id}', 'purchase_quantity': {purchase_quantity}, 'min_quality_score': {min_quality_score}}}")
        
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