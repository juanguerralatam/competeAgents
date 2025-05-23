#!/usr/bin/env python3
import argparse
from compete.simul import Simulation
from compete.scene import MarketEnvironment
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run the Vendor-ISP Market Simulation')
    parser.add_argument('exp_name', type=str, help='Name of the experiment')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--duration', type=int, default=6, 
                       help='Duration of simulation in months (default: 3)')
    args = parser.parse_args()

    # Set logging level based on debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Validate duration
    if args.duration < 1:
        logging.error("Duration must be at least 1 month")
        return

    # Initialize market environment
    market = MarketEnvironment()
    
    # Create and run simulation
    logging.info(f"Starting simulation '{args.exp_name}' for {args.duration} months")
    simulation = Simulation(market, duration=args.duration, exp_name=args.exp_name)
    simulation.run()
    
    # Log completion
    logging.info(f"Simulation '{args.exp_name}' completed successfully after {args.duration} months")
    logging.info(f"ISP decisions have been saved to logs/isp_decisions_{args.exp_name}_*.csv")

if __name__ == "__main__":
    main() 