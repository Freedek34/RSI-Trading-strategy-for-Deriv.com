import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

# Simulate market data (e.g., random walk for prices)
def generate_market_data(size=1000, timeframe='5min'):
    # Random walk for market prices
    prices = np.cumsum(np.random.randn(size)) + 100  # Starting price at 100
    times = pd.date_range(start='2023-01-01', periods=size, freq=timeframe)
    return pd.Series(prices, index=times, name="price")

# Dynamic RSI function with smoothing
def calculate_dynamic_rsi(prices, base_period=14, smoothing_period=5):
    # Calculate volatility (standard deviation)
    volatility = prices.pct_change().rolling(window=base_period).std()

    # Adjust the RSI period based on volatility
    adjusted_period = base_period / (volatility * 100 + 1)
    adjusted_period = adjusted_period.fillna(base_period)  # Fill NaN values with base_period

    # Calculate RSI with dynamically adjusted periods and smoothing
    rsi_values = []
    for i in range(len(prices)):
        if i < base_period - 1:
            rsi_values.append(np.nan)  # Not enough data to calculate RSI
        else:
            period = max(2, int(adjusted_period.iloc[i]))  # Ensure the period is at least 2
            rsi = RSIIndicator(close=prices[:i+1], window=period).rsi().iloc[-1]
            rsi_values.append(rsi)
    
    rsi_series = pd.Series(rsi_values, index=prices.index)
    
    # Apply smoothing to the RSI
    rsi_smoothed = rsi_series.rolling(window=smoothing_period).mean()
    
    return rsi_smoothed

# Digit-based trading strategy with bet amount and Martingale
def digit_based_rsi_strategy(prices, rsi, initial_bet=100, martingale_multiplier=2):
    balance = 10000
    position = None  # "even" or "odd"
    bet_amount = initial_bet
    history = []
    wins = 0
    losses = 0

    for i in range(1, len(prices)):
        if pd.isna(rsi.iloc[i]):
            continue  # Skip if RSI is not available

        rsi_value = round(rsi.iloc[i], 1)
        last_digit = int(str(rsi_value)[-1])

        if last_digit % 2 == 0 and position != "even":  # Even trade signal
            position = "even"
            if balance >= bet_amount:
                balance += bet_amount  # Assume win for even trade
                wins += 1
                bet_amount = initial_bet  # Reset bet amount after win
            else:
                balance -= bet_amount  # Assume loss for even trade
                losses += 1
                bet_amount *= martingale_multiplier  # Apply Martingale
            history.append(("TRADE EVEN", prices.index[i], prices.iloc[i], balance, bet_amount))
        elif last_digit % 2 != 0 and position != "odd":  # Odd trade signal
            position = "odd"
            if balance >= bet_amount:
                balance += bet_amount  # Assume win for odd trade
                wins += 1
                bet_amount = initial_bet  # Reset bet amount after win
            else:
                balance -= bet_amount  # Assume loss for odd trade
                losses += 1
                bet_amount *= martingale_multiplier  # Apply Martingale
            history.append(("TRADE ODD", prices.index[i], prices.iloc[i], balance, bet_amount))

        history.append(("HOLD", prices.index[i], prices.iloc[i], balance, bet_amount))

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    return history, balance, win_rate

# Main function
def main():
    # Generate market data with a specified timeframe
    prices = generate_market_data(1000, timeframe='5min')  # 5-minute timeframe

    # Calculate dynamic RSI with smoothing
    rsi = calculate_dynamic_rsi(prices)

    # Run the digit-based RSI trading strategy with Martingale
    history, final_balance, win_rate = digit_based_rsi_strategy(prices, rsi, initial_bet=100, martingale_multiplier=2)

    # Print final results
    print(f"Initial Balance: $10000")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Plot the results
    history_df = pd.DataFrame(history, columns=["Action", "Date", "Price", "Balance", "Bet Amount"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices, label="Price")
    plt.plot(rsi.index, rsi, label="Smoothed Dynamic RSI", color="orange")
    
    even_trades = history_df[history_df["Action"] == "TRADE EVEN"]
    odd_trades = history_df[history_df["Action"] == "TRADE ODD"]
    plt.scatter(even_trades["Date"], even_trades["Price"], marker="^", color="blue", label="Trade Even", alpha=1)
    plt.scatter(odd_trades["Date"], odd_trades["Price"], marker="v", color="red", label="Trade Odd", alpha=1)
    
    plt.legend()
    plt.title("Digit-Based RSI Trading Strategy with Martingale")
    plt.xlabel("Time")
    plt.ylabel("Price / RSI")
    plt.show()

if __name__ == "__main__":
    main()
