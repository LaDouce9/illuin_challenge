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

### Option 1: Docker (Recommended)

The easiest way to run the project is using Docker with Jupyter Lab.

#### Build the Docker image

```bash
docker build -t illuin-challenge .
```

#### Run Jupyter Lab

**Linux / macOS:**
```bash
docker run -d \
  --name illuin-jupyter \
  -p 8888:8888 \
  -v "$(pwd)":/app \
  illuin-challenge
```

**Windows (PowerShell):**
```powershell
docker run -d `
  --name illuin-jupyter `
  -p 8888:8888 `
  -v "${PWD}:/app" `
  illuin-challenge
```

**Windows (Command Prompt):**
```cmd
docker run -d ^
  --name illuin-jupyter ^
  -p 8888:8888 ^
  -v "%cd%:/app" ^
  illuin-challenge
```

#### Access Jupyter Lab

Open your browser and go to:
```
http://localhost:8888/lab?token=illuin2024
```

**Token:** `illuin2024`

#### Stop the container

```bash
docker stop illuin-jupyter
```

#### Restart the container

```bash
docker start illuin-jupyter
```

#### Remove the container

```bash
docker stop illuin-jupyter
docker rm illuin-jupyter
```

#### Using Makefile (Linux / macOS / WSL)

If you have `make` available, you can use these shortcuts:

```bash
# Build the image
make build

# Run Jupyter Lab
make run-jupyter

# Stop the container
make stop

# Restart the container
make restart

# Clean up (stop and remove container)
make clean
```

### Option 2: Local Python Environment

#### Notebooks

To explore the data, you can use the provided Jupyter notebooks.

1.  Open the project in VS Code.
2.  Open `notebooks/0_prise_en_main.ipynb`.
3.  Select the kernel corresponding to the `.venv` environment.
4.  Run the cells.

Alternatively, you can run scripts using `uv run`:

```bash
uv run python main.py
```