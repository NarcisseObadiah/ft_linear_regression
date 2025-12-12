import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

import predic

#  PRINTING HELPERS

def print_result(theta0_norm, theta1_norm, theta0_denorm, theta1_denorm, cost):
    """Pretty-print the training results."""
    results = [
        ("Theta0_norm", theta0_norm),
        ("Theta1_norm", theta1_norm),
        ("Theta0_denorm", theta0_denorm),
        ("Theta1_denorm", theta1_denorm),
        ("Final Cost", cost),
    ]

    print("-" * 49)
    for label, val in results:
        print(f"{label}\t|\t{val}\t|")
        print("-" * 49)

def print_info():
    """Show menu options."""
    print("\n=== Training Options ===\n")
    print("1. Train the model with raw data")
    print("2. Visualize the regression results")
    print("3. Visualize the cost function over iterations")
    print("4. Quit\n")

#  JSON HANDLING FOR SAVING THETAS
def save_thetas(theta0, theta1):
    """Save θ0 and θ1 to values.json."""
    try:
        with open("values.json", "w") as json_file:
            json.dump({"Theta0": theta0, "Theta1": theta1}, json_file)
    except Exception as e:
        print(f"Error: {e}")


#  DATA NORMALIZATION

def normalize(data):
    """Normalize data between 0 and 1."""
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min

    return [(val - data_min) / data_range for val in data]


def denormalize(theta0_norm, theta1_norm, x_raw, y_raw):
    """
    Convert normalized thetas back to original scale.
    theta0 --> b and theta1 --> w  | --> f(x) = wxi + b
    """
    x_range = max(x_raw) - min(x_raw)
    y_range = max(y_raw) - min(y_raw)

    theta1 = theta1_norm * (y_range / x_range)
    theta0 = theta0_norm * y_range + min(y_raw) - theta1 * min(x_raw)

    return float(theta0), float(theta1)

#  GRADIENT DESCENT CORE
def error(theta1, theta0, x, y):
    """Compute prediction error: (θ0 + θ1 * x) - y."""
    return (theta0 + theta1 * x) - y

#  GRADIENT UPDATE RULES (STEP-BY-STEP)

def gradient_theta1(theta1, theta0, x_data, y_data):
    """
    Gradient of cost function with respect to θ1.
    Formula:
        dJ/dθ1 = (1/m) * Σ[ (θ0 + θ1*x_i - y_i) * x_i ]
    """
    m = len(x_data)
    sum_errors = 0

    for x, y in zip(x_data, y_data):
        sum_errors += error(theta1, theta0, x, y) * x
    return sum_errors / m


def gradient_theta0(theta1, theta0, x_data, y_data):
    """
    Gradient of cost function with respect to θ0.
    Formula:
        dJ/dθ0 = (1/m) * Σ[ (θ0 + θ1*x_i - y_i) ]
    """
    m = len(x_data)
    sum_errors = 0

    for x, y in zip(x_data, y_data):
        sum_errors += error(theta1, theta0, x, y)
    return sum_errors / m


def apply_gradient_descent(theta1, theta0, x_data, y_data, learning_rate=0.1):
    """
    Apply one iteration of gradient descent:
    
        θ0 := θ0 - lr * dJ/dθ0
        θ1 := θ1 - lr * dJ/dθ1
        whith  lr as learning rate
    """
    grad0 = gradient_theta0(theta1, theta0, x_data, y_data)
    grad1 = gradient_theta1(theta1, theta0, x_data, y_data)

    theta0 -= learning_rate * grad0
    theta1 -= learning_rate * grad1

    return theta0, theta1

#  COST FUNCTION + TRAINING LOOP
def squared_error(theta1, theta0, x_data, y_data):
    """Compute total squared error."""
    total = 0
    for x, y in zip(x_data, y_data):
        total += error(theta1, theta0, x, y) ** 2
    return total

def cost(theta1, theta0, x_data, y_data):
    """Compute full cost function."""
    m = len(x_data)
    return (1 / (2 * m)) * squared_error(theta1, theta0, x_data, y_data)

def train_model(theta1, theta0, x_data, y_data, iterations=1000):
    """Run full gradient descent training."""
    cost_history = []

    for _ in range(iterations):
        curr_cost = cost(theta1, theta0, x_data, y_data)
        cost_history.append(curr_cost)
        theta0, theta1 = apply_gradient_descent(theta1, theta0, x_data, y_data)
    return theta0, theta1, iterations, cost_history

#  VISUALIZATION

def visualize_regression(theta0, theta1, x_data, y_data):
    """Scatter + regression line."""
    # x = np.array(x_data)
    # y = np.array(y_data)
    y_pred = predic.estimate_price(x_data, theta0, theta1)

    plt.scatter(x_data, y_data)
    plt.plot(x_data, y_pred, color="red")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (Euro)")
    # plt.show()
    plt.savefig('plot.png') 

def visualize_cost(iterations, cost_history):
    plt.plot(range(iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.savefig('cost_plot.png')
    # plt.show()

#  INPUT HELPERS
def get_input_choice():
    try:
        value = int(input("Proceed: "))
        return value
    except ValueError:
        print("Invalid number")
        return None

def main():
    theta0 = 0
    theta1 = 0

    try:
        data = pd.read_csv("data.csv")
        x_raw = np.array(data["km"])
        y_raw = np.array(data["price"])

        x_norm = normalize(x_raw)
        y_norm = normalize(y_raw)

        print_info()
        choice = get_input_choice()
        os.system("clear")

        # Training always happens
        theta0_n, theta1_n, iters, cost_hist = train_model(theta1, theta0, x_norm, y_norm)
        theta0_d, theta1_d = denormalize(theta0_n, theta1_n, x_raw, y_raw)

        if choice == 1:
            print_result(theta0_n, theta1_n, theta0_d, theta1_d, cost_hist[-1])
        elif choice == 2:
            visualize_regression(theta0_d, theta1_d, x_raw, y_raw)
        elif choice == 3:
            visualize_cost(iters, cost_hist)
        elif choice == 4:
            return
        else:
            print("Enter a valid number between 1–4")
        save_thetas(theta0_d, theta1_d)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
