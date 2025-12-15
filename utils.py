import matplotlib.pyplot as plt
import json
import predic

#display functions

def display_result(theta0_norm, theta1_norm, theta0, theta1, final_cost, r2_score_value):
    """Print training summary with theta parameters."""
    print("\nTraining summary:")
    print(f"- theta0 (normalized): {theta0_norm:.6f}")
    print(f"- theta1 (normalized): {theta1_norm:.6f}")
    print(f"- theta0:              {theta0:.6f}")
    print(f"- theta1:              {theta1:.6f}")
    print(f"- final cost:          {final_cost:.6f}")
    print(f"- R2 score:            {r2_score_value:.6f}\n")


def show_menu():
    """Show available actions."""
    print("\nSelect an option:")
    print(" 1) Train on data")
    print(" 2) Show regression plot")
    print(" 3) Show cost plot")
    print(" 4) Quit\n")

#file operations

def save_thetas(theta0, theta1):
    """Save θ0 and θ1 to thetas.json."""
    try:
        with open("thetas.json", "w") as json_file:
            json.dump({"theta0": theta0, "theta1": theta1}, json_file)
    except Exception as e:
        print(f"Error: {e}")


# Normalization

def normalize(data):
    """Normalize data between 0 and 1"""
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min

    return [(val - data_min) / data_range for val in data]


def denormalize(theta0_norm, theta1_norm, mileage_data, price_data):
    """
    Convert normalized thetas back to original scale.
    f(x) = theta1 * x + theta0
    """
    mileage_range = max(mileage_data) - min(mileage_data)
    price_range = max(price_data) - min(price_data)

    theta1 = theta1_norm * (price_range / mileage_range)
    theta0 = theta0_norm * price_range + min(price_data) - theta1 * min(mileage_data)

    return float(theta0), float(theta1)



# visualization

def display_regression_plot(theta0, theta1, mileage_data, price_data):
    predicted_prices = predic.estimate_price(mileage_data, theta0, theta1)

    plt.scatter(mileage_data, price_data)
    plt.plot(mileage_data, predicted_prices, color="red")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (Euro)")
    plt.savefig("plot.png")


def display_cost_plot(iterations, cost_history):
    plt.plot(range(iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.savefig("cost_plot.png")


# precision

def r2_score(actual_prices, predicted_prices):
    mean_price = sum(actual_prices) / len(actual_prices)

    residual_sum_squares = 0
    for real_price, predicted_price in zip(actual_prices, predicted_prices):
        residual_sum_squares += (real_price - predicted_price) ** 2

    total_sum_squares = 0
    for real_price in actual_prices:
        total_sum_squares += (real_price - mean_price) ** 2

    return 1 - (residual_sum_squares / total_sum_squares)


def model_precision(theta0, theta1, mileage_data, price_data):
    predicted_prices = []
    for mileage in mileage_data:
        predicted_prices.append(theta0 + theta1 * mileage)

    return r2_score(price_data, predicted_prices)


# input handling

def get_input_choice():
    try:
        return int(input("Proceed: "))
    except ValueError:
        print("Invalid number")
        return None
