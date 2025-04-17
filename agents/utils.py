import json
import logging
import os
from datetime import datetime

class LoggingUtils:
    """
    A utility class for logging conversations and tool usage in agent interactions.
    """
    
    @staticmethod
    def setup_logs_directory():
        """Create logs directory if it doesn't exist"""
        os.makedirs('logs', exist_ok=True)
        
    @classmethod
    def get_log_filenames(cls, agent_id: str) -> tuple:
        """Generate log filenames for an agent"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_log = f'logs/{agent_id}_conversations_{timestamp}.txt'
        tool_usage_log = f'logs/{agent_id}_tool_usage_{timestamp}.txt'
        return conversation_log, tool_usage_log
        
    @staticmethod
    def log_conversation(log_file: str, agent_id: str, system_prompt: str, decision_prompt: str, response: str):
        """Log the full conversation to a file"""
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Agent: {agent_id}\n")
            f.write(f"\nSystem Prompt:\n{system_prompt}\n")
            f.write(f"\nDecision Prompt:\n{decision_prompt}\n")
            f.write(f"\nLLM Response:\n{response}\n")
            f.write(f"{'='*50}\n")
        
        # Also log to the main logger
        logging.info(f"\n{'='*50}")
        logging.info(f"Agent {agent_id} LLM Conversation:")
        logging.info(f"System Prompt:\n{system_prompt}")
        logging.info(f"Decision Prompt:\n{decision_prompt}")
        logging.info(f"LLM Response:\n{response}")
        logging.info(f"{'='*50}\n")
        
    @staticmethod
    def log_tool_usage(log_file: str, agent_id: str, tool_name: str, input_data: dict, output_data: dict):
        """Log tool usage to a file"""
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Agent: {agent_id}\n")
            f.write(f"Tool: {tool_name}\n")
            f.write(f"Input: {json.dumps(input_data, indent=2)}\n")
            f.write(f"Output: {json.dumps(output_data, indent=2)}\n")
            f.write(f"{'='*50}\n")
            
        # Also log to the main log
        logging.info(f"Agent {agent_id} used tool {tool_name}")
        logging.debug(f"Tool input: {json.dumps(input_data, indent=2)}")
        logging.debug(f"Tool output: {json.dumps(output_data, indent=2)}")