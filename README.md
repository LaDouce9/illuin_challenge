# Illuin Challenge

This project contains the code and data for the Illuin Challenge.

## Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) for dependency management.

## Installation

### 1. Install `uv`

You need to install `uv` to manage the project dependencies.

**Windows:**

You can install `uv` using pip:
```bash
pip install uv
```

Or via PowerShell (if allowed by your security policy):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

Use the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment

Once `uv` is installed, synchronize the project dependencies:

```bash
uv sync
```

This will create a virtual environment in `.venv` with all the required packages.

## Usage

### Notebooks

To explore the data, you can use the provided Jupyter notebooks.

1.  Open the project in VS Code.
2.  Open `notebooks/0_prise_en_main.ipynb`.
3.  Select the kernel corresponding to the `.venv` environment.
4.  Run the cells.

Alternatively, you can run scripts using `uv run`:

```bash
uv run python main.py
```