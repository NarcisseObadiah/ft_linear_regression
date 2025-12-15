import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os

from utils import (
    display_result,
    show_menu,
    save_thetas,
    normalize,
    denormalize,
    display_regression_plot,
    display_cost_plot,
    model_precision,
    get_input_choice,
)


# Core training helpers

def error(theta1, theta0, mileage, price):
    """Compute prediction error: (theta0 + theta1 * x) - y."""
    return (theta0 + theta1 * mileage) - price


def gradient_theta1(theta1, theta0, mileage_data, price_data):
    m = len(mileage_data)
    total = 0
    for x, y in zip(mileage_data, price_data):
        total += error(theta1, theta0, x, y) * x
    return total / m


def gradient_theta0(theta1, theta0, mileage_data, price_data):
    m = len(mileage_data)
    total = 0
    for x, y in zip(mileage_data, price_data):
        total += error(theta1, theta0, x, y)
    return total / m


def apply_gradient_descent(theta1, theta0, mileage_data, price_data, learning_rate=0.1):
    grad0 = gradient_theta0(theta1, theta0, mileage_data, price_data)
    grad1 = gradient_theta1(theta1, theta0, mileage_data, price_data)

    theta0 -= learning_rate * grad0
    theta1 -= learning_rate * grad1

    return theta0, theta1


def squared_error(theta1, theta0, mileage_data, price_data):
    total = 0
    for x, y in zip(mileage_data, price_data):
        total += error(theta1, theta0, x, y) ** 2
    return total


def cost(theta1, theta0, mileage_data, price_data):
    m = len(mileage_data)
    return (1 / (2 * m)) * squared_error(theta1, theta0, mileage_data, price_data)


def train_model(theta1, theta0, mileage_data, price_data, iterations=1000):
    cost_history = []

    for _ in range(iterations):
        cost_history.append(cost(theta1, theta0, mileage_data, price_data))
        theta0, theta1 = apply_gradient_descent(theta1, theta0, mileage_data, price_data)

    return theta0, theta1, iterations, cost_history


def load_data():
    """Load and validate data from CSV file."""
    try:
        data = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("Error: data.csv not found.")
        return None, None
    except Exception as e:
        print(f"Error reading data.csv: {e}")
        return None, None

    if "km" not in data.columns or "price" not in data.columns:
        print("Error: data.csv must contain 'km' and 'price' columns.")
        return None, None

    mileage_data = np.array(data["km"])
    price_data = np.array(data["price"])

    if len(mileage_data) == 0:
        print("Error: No data available for training.")
        return None, None

    return mileage_data, price_data


def main():
    theta0 = 0
    theta1 = 0

    try:
        mileage_data, price_data = load_data()
        if mileage_data is None:
            return

        mileage_norm = normalize(mileage_data)
        price_norm = normalize(price_data)

        show_menu()
        choice = get_input_choice()
        
        if choice is None:
            return
        
        os.system("clear")

        theta0_n, theta1_n, iteration_count, cost_history = train_model(theta1, theta0, mileage_norm, price_norm)
        theta0_d, theta1_d = denormalize( theta0_n, theta1_n, mileage_data, price_data)
        r2_precision = model_precision(theta0_d, theta1_d, mileage_data, price_data)

        if choice == 1:
            display_result(
                theta0_n,
                theta1_n,
                theta0_d,
                theta1_d,
                cost_history[-1],
                r2_precision,
            )
        elif choice == 2:
            display_regression_plot(theta0_d, theta1_d, mileage_data, price_data)
        elif choice == 3:
            display_cost_plot(iteration_count, cost_history)
        elif choice == 4:
            return
        else:
            print("Enter a valid number between 1-4")

        save_thetas(theta0_d, theta1_d)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
