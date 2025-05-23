from typing import Dict, List, Any
from agents.agent import VendorAgent, ISPAgent
import json
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os

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
        """Initialize vendor and ISP agents with defined parameters"""
        logging.info("Initializing agents...")
        # Initialize vendors
        vendor1 = VendorAgent(
            "vendor1",
            {
                'brand': 'Vendor1',
                'fix_cost': 100000,
                'variable_cost': 50000,
                'capital': 1000000,
                'cash_flow': 50000,
                'salary_rd': 50000,
                'salary_maketing': 30000,
                'products': [
                    {
                        'name': 'Basic ONU',
                        'price': 100,
                        'cost': 60,
                        'description': 'Entry-level ONU device'
                    }
                ],
                'content': 'Basic marketing campaign',
                'income': 0,
                'expenses': 0,
                'sales': 0,
                'Updated_Yearly': True,
                'rival_info': {},
                'plan': 'cost_leadership',
                'analysis_portfolios': 0.5,
                'score_product': 0.5,
                'score_add': 0.5
            }
        )
        
        vendor2 = VendorAgent(
            "vendor2",
            {
                'brand': 'Vendor2',
                'fix_cost': 120000,
                'variable_cost': 60000,
                'capital': 1200000,
                'cash_flow': 60000,
                'salary_rd': 60000,
                'salary_maketing': 40000,
                'products': [
                    {
                        'name': 'Premium ONU',
                        'price': 200,
                        'cost': 120,
                        'description': 'High-end ONU device'
                    }
                ],
                'content': 'Premium marketing campaign',
                'income': 0,
                'expenses': 0,
                'sales': 0,
                'Updated_Yearly': True,
                'rival_info': {},
                'plan': 'cost_focus',
                'analysis_portfolios': 0.7,
                'score_product': 0.7,
                'score_add': 0.7
            }
        )
        self.vendors = [vendor1, vendor2]
        
        # Initialize ISPs
        isp1 = ISPAgent(
            "isp1",
            {
                'cash_flow': 50000,
                'buy': 0,
                'analisys_porfolios': 0,
                'score_product': 0,
                'score_add': 0
            }
        )
        
        isp2 = ISPAgent(
            "isp2",
            {
                'cash_flow': 75000,
                'buy': 0,
                'analisys_porfolios': 0,
                'score_product': 0,
                'score_add': 0
            }
        )
        self.isps = [isp1, isp2]
        logging.info(f"Initialized {len(self.vendors)} vendors and {len(self.isps)} ISPs")
        
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
        """Apply ISP decision and log it"""
        # Log and store decision
        self._log_isp_decision(isp.agent_id, decision)
        isp.state.current_state['buy'] = decision['purchase_quantity']
        isp.state.current_state['min_quality_score'] = decision['min_quality_score']
        # Update vendor sales
        selected_vendor = next(v for v in self.vendors if v.agent_id == decision['selected_vendor'])
        selected_vendor.state.current_state['sales'] += decision['purchase_quantity']
        logging.info(f"ISP {isp.agent_id} made decision: {json.dumps(decision, indent=2)}")
        
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