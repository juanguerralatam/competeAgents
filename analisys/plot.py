import matplotlib.pyplot as plt

# Graph 1: Price Comparison: Cost Leadership vs Cost Differentiation

months = [f"Month {i+1}" for i in range(12)]
company_a_prices = [50] * 12  # Constant price for Cost Leadership (Company A)
company_b_prices = [100 - 5*i for i in range(12)]  # Decreasing price for Cost Differentiation (Company B)

plt.figure(figsize=(10, 6))
plt.plot(months, company_a_prices, label="Company A (Cost Leadership)", marker='o', color='blue')
plt.plot(months, company_b_prices, label="Company B (Cost Differentiation)", marker='o', color='green')
plt.title("Price Comparison: Cost Leadership vs Cost Differentiation", fontsize=16)
plt.xlabel("Months", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("price_comparison_cost_strategies.png", dpi=300)
plt.show()

# Graph 2: Order vs Time (Flat orders over time)
orders = [500] * 12  # Stable order quantity for 12 months

plt.figure(figsize=(10, 6))
plt.plot(months, orders, label="Orders", color='black')
plt.title("Order vs Time", fontsize=16)
plt.xlabel("Months", fontsize=12)
plt.ylabel("Orders", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("order_vs_time.png", dpi=300)
plt.show()

# Graph 3: Vendors vs Categories (Bar chart with 3 categories)
categories = ['H', 'M', 'L']
values = [300, 450, 200]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['blue', 'green', 'red'])
plt.title("Vendors vs Categories", fontsize=16)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Vendors", fontsize=12)
plt.tight_layout()
plt.savefig("vendors_vs_categories.png", dpi=300)
plt.show()
