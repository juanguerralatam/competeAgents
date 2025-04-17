# Competitive Vendor-ISP Market Simulation

A multi-agent based simulation (ABM) modeling the competitive dynamics between ONU (Optical Network Unit) device manufacturers (vendors) and Internet Service Providers (ISPs) using LangGraph and DeepSeek for intelligent agent behavior.

## Project Overview

This simulation creates a dynamic market environment featuring:

- **2 Vendor Agents:** Representing manufacturers of ONU devices.
- **10 ISP Agents:** Representing buyers of ONU devices.

The simulation models how vendors compete for market share by implementing various business strategies, while ISPs make purchasing decisions based on a range of criteria and market conditions.

### Key Simulation Features

- **Intelligent Agent Behavior:**
    - **Vendors:**
        - Manufacture diverse ONU devices with varying characteristics.
        - Employ and adapt competitive strategies:
            - **Cost Leadership:** Focusing on high-volume production and economies of scale.
            - **Cost Focus:** Targeting specific market niches with specialized products.
        - Adjust pricing and product strategies dynamically based on their current market share and competitive landscape.
        - Track and utilize a summary of the previous three simulation steps for informed decision-making.
    - **ISPs:**
        - Make monthly purchasing decisions based on their budget constraints.
        - Evaluate vendor offerings based on multiple criteria (e.g., price, quality, features).
        - Maintain a history of their past purchase decisions.
        - Analyze vendor portfolios to inform their selection process.
        - Consider product quality and added services (score_product, score_add) in their evaluations.
        - Utilize a summary of the previous three simulation steps for informed decision-making.

### Technical Stack

- **Core AI & Orchestration:**
    - Python 3.11
    - LangGraph: For orchestrating the interactions and workflows of multiple agents.
    - DeepSeek: As the language model powering agent decision-making.
    - Ollama API (accessible at `http://localhost:11434/api/generate`): For local inference of the language model.
    - Django with Langchain: For managing the simulation database.
- **Data Handling & Analysis:**
    - Pandas: For efficient data manipulation and analysis.
    - NumPy: For numerical computations.
    - Matplotlib: For creating visualizations of simulation results.
- **Configuration & Logging:**
    - JSON: For logging agent decisions, state changes, and transaction records.
    - CSV: For storing time-series data and analytics.
    - YAML/JSON Parsers: For handling configuration files.
    - Django: For other relevant information

## Project Structure

```
project/
├── agents/         # Contains agent definitions and logic
│   ├── agent.py
│   └── tools.py      # Custom tools agents can use
├── compete/        # Core simulation logic
│   ├── simul.py    # Main simulation engine
│   └── scene.py    # Defines the simulation environment and setup
├── strategies/     # Vendor strategy implementations
│   ├── market_analysis.py # Tools for analyzing the market
│   └── plan.py          # Logic for generating strategic plans
├── data/           # Data storage and output
│   ├── database.sql  # Simulation database (managed by Django)
│   ├── logs/         # JSON logs of simulation events
│   └── analytics/    # CSV files for analyzed data
├── config/         # Configuration files
│   ├── global_params.json # Global simulation settings
│   └── ...           # Other configuration files
├── main.py         # Script to run the simulation
├── utils.py        # Utility functions
└── requirements.txt # Project dependencies
```
## Simulation Parameters

### Global Parameters (Updated Annually)
- **Company Basics:** `[brand, fix_cost, variable_cost, capital, cash_flow]` (Initial information for all companies)
- **Environmental Constants:** `[country(H, M, L), city(1, 2), company_size(L, S)]` (Categorical factors influencing the market)
- **Market Growth:** `[internet_grown, economy_grown]` (Annual percentage increase in these sectors)
- **Technological Advancement:** `[migration_fiber]` (Annual rate of fiber optic adoption)

### Vendor Parameters
- **Investments:** `[salary_rd]` (R&D budget), `[salary_maketing]` (Marketing budget)
- **Product Portfolio:** `[name, price, cost, description]` (Details of manufactured ONU devices)
- **Marketing Campaigns:** `[content]` (Content of active marketing efforts)
- **Operations:** `[income, expenses, sales]` (Financial performance metrics)
- **Strategy:** `[Updated_Yearly, rival_info, plan]` (Annual strategy update, information about competitors, strategic plan)
- **Memory:** Summary of the previous 3 simulation steps for strategic context.

### ISP Parameters
- **Financials:** `[cash_flow, buy]` (Available budget, planned purchase quantity)
- **Selection Criteria:** `[analisys_porfolios]` (Results of portfolio analysis for vendor selection)
- **Quality Assessment:** `[score_product, score_add]` (Scores for product quality and added services)
- **Memory:** Summary of the previous 3 simulation steps for purchase decisions.

## Key Milestones

1. **Strategy Implementation (Vendors):**
    - **Cost Leadership Strategy:**
        - Focus on operational efficiency to minimize production costs.
        - Employ market penetration pricing to gain market share.
    - **Cost Focus Strategy:**
        - Develop specialized products tailored to specific market segments.
        - Implement premium pricing strategies for these niche segments.

2. **Market Share Analysis:**
    - Track and analyze vendor performance metrics (e.g., sales volume, revenue).
    - Monitor the distribution of market share among competing vendors over time.

## Data Output

## Memory
- Brand Company name Vendor

### JSON Logs
- Detailed records of agent decision-making processes.
- Logs of strategic changes implemented by vendors.
- Records of significant market events.
- Transaction details between ISPs and vendors.
- Specifications and attributes of products offered.
- Content and details of marketing campaigns.

### CSV Analytics
- Time-series data showing market share trends for each vendor.
- Evolution of product pricing over the simulation period.
- Patterns in ISP purchasing behavior (vendor preferences, quantities).
- Metrics evaluating the performance of different vendor strategies.

## Getting Started

1. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Modify global simulation parameters in `config/global_params.json`
   - Adjust initial agent strategies and behaviors in the respective strategy files within the `strategies/` directory

3. **Running the Simulation**
   ```bash
   python main.py exp_name=compete
   ```

## Analysis Tools

-Scripts for visualizing market share dynamics using Matplotlib.
-Tools for analyzing the performance of different vendor strategies.
-Methods for identifying recurring decision patterns in agent behavior.
-Framework for evaluating the overall competitive dynamics of the simulated market.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 