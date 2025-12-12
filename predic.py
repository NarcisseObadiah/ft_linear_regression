import json
import sys

# Load saved thetas
def load_thetas(filename="values.json"):
    """
    Load Theta0 and Theta1 from JSON file produced by training.
    These values are already denormalized, so they accept raw mileage.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'values.json' not found. Run training first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: 'values.json' contains invalid JSON.")
        sys.exit(1)

    # Validate keys
    try:
        theta0 = float(data["Theta0"])
        theta1 = float(data["Theta1"])
    except KeyError:
        print("Error: Missing keys 'Theta0' or 'Theta1' in values.json.")
        sys.exit(1)
    except (TypeError, ValueError):
        print("Error: Invalid numeric data in values.json.")
        sys.exit(1)

    return theta0, theta1


# prediction logic-- > f(x) = Wxi + B
def estimate_price(mileage, theta0, theta1):
    """price = theta0 + theta1 * mileage"""
    return theta0 + theta1 * mileage

if __name__ == "__main__":
    theta0, theta1 = load_thetas()

    try:
        mileage = float(input("Enter car mileage (km): "))
    except ValueError:
        print("Invalid mileage input.")
        sys.exit(1)

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f} €")
