import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
# from rich.layout import Layout
import torch
from chibigrad.tensor import Tensor
from chibigrad.linear import Linear
from chibigrad.loss import MSELoss
import numpy as np

"""
python check.py --test arithmetic  # Basic operations
python check.py --test mse         # MSE loss tests
python check.py --test linear      # Linear layer tests
"""

console = Console()


def format_number(x, precision=4):
    """Format number to specified decimal places"""
    if isinstance(x, (int, float)):
        return f"{x:.{precision}f}"
    return [f"{n:.{precision}f}" for n in x]


def format_data(data, precision=4):
    """Format data that could be scalar or array"""
    if isinstance(data, (list, np.ndarray)):
        return [format_number(x, precision) for x in data]
    return format_number(data, precision)


def create_comparison_table(my_values, torch_values, title, threshold=1e-6):
    """Create a comparison table between our implementation and PyTorch"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Value Match", justify="center")
    table.add_column("Gradient Match", justify="center")
    table.add_column("Max Value Diff", justify="right")
    table.add_column("Max Grad Diff", justify="right")

    for key in my_values:
        value_match = (
            "[green]✓[/green]"
            if np.allclose(
                my_values[key]["value"], torch_values[key]["value"], atol=threshold
            )
            else "[red]✗[/red]"
        )

        if my_values[key]["grad"] is not None and torch_values[key]["grad"] is not None:
            grad_match = (
                "[green]✓[/green]"
                if np.allclose(
                    my_values[key]["grad"], torch_values[key]["grad"], atol=threshold
                )
                else "[red]✗[/red]"
            )
            grad_diff = f"{np.max(np.abs(my_values[key]['grad'] - torch_values[key]['grad'])):.2e}"
        else:
            grad_match = "N/A"
            grad_diff = "N/A"

        value_diff = f"{np.max(np.abs(my_values[key]['value'] - torch_values[key]['value'])):.2e}"

        table.add_row(key, value_match, grad_match, value_diff, grad_diff)

    return table


def print_detailed_comparison(name, chibigrad_grad, torch_grad):
    """Print detailed gradient comparison"""
    console.print(f"\nDetailed gradient comparison for {name}:")
    console.print(f"This chibigrad's gradient: {chibigrad_grad}")
    console.print(f"PyTorch's gradient: {torch_grad}")


def test_basic_arithmetic():
    """Test basic arithmetic operations"""
    # Create tensors
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)

    # building computation graph with operation names
    operations = {
        "MatMul (a @ b)": a @ b,
        "Add (c + b)": None,  # Will be set after first op
        "Multiply (d * b)": None,
        "Divide (e / b)": None,
        "Power (f ** b)": None,
    }

    # executing operations sequentially
    c = operations["MatMul (a @ b)"]
    operations["Add (c + b)"] = c + b
    d = operations["Add (c + b)"]
    operations["Multiply (d * b)"] = d * b
    e = operations["Multiply (d * b)"]
    operations["Divide (e / b)"] = e / b
    f = operations["Divide (e / b)"]
    operations["Power (f ** b)"] = f**b
    g = operations["Power (f ** b)"]

    # Backprop
    g.backward()

    # store my values with operation names
    my_values = {
        "a": {"value": a.data, "grad": a.grad},
        "b": {"value": b.data, "grad": b.grad},
    }
    # adding operations to values
    for op_name, tensor in operations.items():
        if tensor is not None:  # Skip None values
            my_values[op_name] = {"value": tensor.data, "grad": tensor.grad}

    # PyTorch comparison
    a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    # executing PyTorch operations
    c_torch = a_torch @ b_torch
    c_torch.retain_grad()
    d_torch = c_torch + b_torch
    d_torch.retain_grad()
    e_torch = d_torch * b_torch
    e_torch.retain_grad()
    f_torch = e_torch / b_torch
    f_torch.retain_grad()
    g_torch = f_torch**b_torch
    g_torch.retain_grad()

    g_torch.backward(torch.ones_like(g_torch))

    # storing PyTorch values with operation names
    torch_values = {
        "a": {"value": a_torch.detach().numpy(), "grad": a_torch.grad.numpy()},
        "b": {"value": b_torch.detach().numpy(), "grad": b_torch.grad.numpy()},
        "MatMul (a @ b)": {
            "value": c_torch.detach().numpy(),
            "grad": c_torch.grad.numpy(),
        },
        "Add (c + b)": {
            "value": d_torch.detach().numpy(),
            "grad": d_torch.grad.numpy(),
        },
        "Multiply (d * b)": {
            "value": e_torch.detach().numpy(),
            "grad": e_torch.grad.numpy(),
        },
        "Divide (e / b)": {
            "value": f_torch.detach().numpy(),
            "grad": f_torch.grad.numpy(),
        },
        "Power (f ** b)": {
            "value": g_torch.detach().numpy(),
            "grad": g_torch.grad.numpy(),
        },
    }

    # printing detailed comparisons
    for op_name in operations:
        if op_name in my_values and op_name in torch_values:
            print_detailed_comparison(
                op_name, my_values[op_name]["grad"], torch_values[op_name]["grad"]
            )

    # creating and display comparison table
    table = create_comparison_table(
        my_values, torch_values, "Arithmetic Operations Comparison"
    )
    console.print("\nComparison Results:")
    console.print(table)


def test_mse_linear():
    """Test MSE loss with linear layer"""
    np.random.seed(42)
    torch.manual_seed(42)

    # my implementation
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y_true = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    linear = Linear(3, 3)
    y_pred = linear(x)
    criterion = MSELoss()
    loss = criterion(y_pred, y_true)
    loss.backward()

    # storing my values
    my_values = {
        "Predictions": {"value": y_pred.data, "grad": y_pred.grad},
        "Weights": {"value": linear.weight.data, "grad": linear.weight.grad},
        "Bias": {"value": linear.bias.data, "grad": linear.bias.grad},
    }

    # PyTorch implementation
    x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y_true_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    linear_torch = torch.nn.Linear(3, 3)
    linear_torch.weight.data = torch.tensor(linear.weight.data)
    linear_torch.bias.data = torch.tensor(linear.bias.data)

    y_pred_torch = linear_torch(x_torch)
    y_pred_torch.retain_grad()

    criterion_torch = torch.nn.MSELoss()
    loss_torch = criterion_torch(y_pred_torch, y_true_torch)
    loss_torch.backward()

    # storing PyTorch values
    torch_values = {
        "Predictions": {
            "value": y_pred_torch.detach().numpy(),
            "grad": (
                y_pred_torch.grad.numpy() if y_pred_torch.grad is not None else None
            ),
        },
        "Weights": {
            "value": linear_torch.weight.detach().numpy(),
            "grad": linear_torch.weight.grad.numpy(),
        },
        "Bias": {
            "value": linear_torch.bias.detach().numpy(),
            "grad": linear_torch.bias.grad.numpy(),
        },
    }

    # printing detailed comparisons
    for name in ["Predictions", "Weights", "Bias"]:
        if (
            my_values[name]["grad"] is not None
            and torch_values[name]["grad"] is not None
        ):
            print_detailed_comparison(
                name, my_values[name]["grad"], torch_values[name]["grad"]
            )

    # creating and displaying comparison table
    table = create_comparison_table(
        my_values, torch_values, "Linear Layer + MSE Comparison"
    )
    console.print("\nComparison Results:")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Autograd Engine Tests",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test",
        choices=["arithmetic", "mse", "all"],
        nargs="?",  # making the argument optional
        default="all",
        help="""Type of test to run (default: all)
arithmetic: Basic arithmetic operations
mse: MSE loss with linear layer
all: Run all tests""",
    )

    try:
        args = parser.parse_args()

        if args.test == "all":
            # running all tests
            console.print(Panel.fit("Running ALL tests", style="bold magenta"))
            console.print("\n[bold cyan]Running Arithmetic tests:[/bold cyan]")
            test_basic_arithmetic()
            console.print("\n[bold cyan]Running MSE tests:[/bold cyan]")
            test_mse_linear()
        else:
            # printing header for specific test
            console.print(
                Panel.fit(f"Running {args.test.upper()} tests", style="bold magenta")
            )

            if args.test == "arithmetic":
                test_basic_arithmetic()
            elif args.test == "mse":
                test_mse_linear()

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        console.print("\n[yellow]Usage examples:[/yellow]")
        console.print("python -m tests.check           # Run all tests")
        console.print("python -m tests.check --test all        # Run all tests")
        console.print("python -m tests.check --test arithmetic # Run arithmetic tests")
        console.print("python -m tests.check --test mse        # Run MSE tests")
        parser.exit(1)


if __name__ == "__main__":
    main()
