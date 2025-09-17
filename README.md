## MiniTransformer: A Minimalist Transformer for Small-Sample Sequential Data

A PyTorch implementation of the MiniTransformer, a compact Transformer architecture designed for small-sample clinical and behavioral data. The model balances predictive performance with interpretability by combining architectural simplifications with a built-in framework for statistical testing.

![Main Results](main_figure.png)

## Overview

This project implements a custom transformer architecture (`MiniTransformer`) that learns patterns in sequential data, particularly focusing on:

- **Feature Interactions**: Understanding how different features at various positions influence predictions
- **Statistical Testing**: Rigorous evaluation of learned patterns with significance testing
- **Context Effects**: Analysis of how historical context affects future predictions

## Architecture

The `MiniTransformer` consists of:

- **Multi-Head Attention**: Custom attention mechanism with positional encodings
- **Distance Matrices**: Incorporates both pairwise distances and distance-to-end information
- **Custom Masking**: Specialized attention masks for prediction tasks
- **Statistical Analysis**: Built-in tools for evaluating feature importance and interactions

## Key Features

### 1. Data Handling
- **Simulated Data**: Generates synthetic sequential data with controllable patterns
- **Real Data**: Supports real-world datasets (GHQ health questionnaire data)
- **Variable Length Sequences**: Handles sequences of different lengths with proper padding

### 2. Model Components
- Multi-head attention with customizable heads, key/value dimensions
- Position-aware attention using distance matrices
- Cumulant-based feature aggregation
- L2 regularization with bias exclusion

### 3. Evaluation & Analysis
- Multiple baseline comparisons (average, informed, repeat baselines)
- Regression baseline using scikit-learn
- Statistical significance testing with p-value computation
- Context-target effect visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mini_transformer
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run the main training script:

```bash
python main.py
```

### Configuration

Key hyperparameters can be modified in `main.py`:

```python
# Data configuration
data_str = "simulation"  # or "ghq_sum", "ghq_b_sum"
batch_size = 1
n = 200                  # Training samples
p = 10                   # Number of features
maxlen = 10              # Maximum sequence length

# Model architecture
nheads = 16              # Number of attention heads
ncum = 2                 # Number of cumulants
dk = 1                   # Key dimension
dv = 1                   # Value dimension

# Training
learning_rate = 1e-3
lambda_l2 = 1e-3
EPOCHS = 100
```

### Jupyter Notebooks

Explore the analysis notebooks in the `notebooks/` directory:

- `simulation_experiments.ipynb`: Basic simulation experiments
- `simulation_experiments_statistical_testing.ipynb`: Statistical analysis of simulated data
- `real_data_experiments_D1.ipynb`: Real data analysis (Dataset 1)
- `real_data_experiments_D2.ipynb`: Real data analysis (Dataset 2)

## Project Structure

```
mini_transformer/
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── transformers.py     # MiniTransformer implementation
│   ├── data_preparation.py # Data loading and preprocessing
│   ├── evaluation.py       # Model evaluation metrics
│   └── statistical_testing.py # Statistical analysis tools
├── notebooks/             # Jupyter notebooks for experiments
│   ├── simulation_experiments.ipynb
│   ├── real_data_experiments_D1.ipynb
│   └── ...
└── runs/                  # TensorBoard logs and results
```

## Key Components

### MiniTransformer Class

The core model implementing:
- Multi-head attention with distance-aware weights
- Custom masking for causal prediction
- Linear prediction layer

### Data Generation

- `SimulatedDataset`: Generates synthetic sequential data with controlled dependencies
- Variable sequence lengths with probabilistic termination
- Configurable feature interactions

### Statistical Testing

- Permutation-based significance testing
- Context-target effect analysis
- P-value computation with multiple comparisons correction
- Visualization of feature interactions

## Results

The model evaluation includes:

1. **Loss Comparisons**: Against multiple baselines (average, informed, regression)
2. **Statistical Significance**: P-values for feature interactions
3. **Context Effects**: Heatmaps showing how context influences predictions
4. **Parameter Analysis**: Distance weights and attention patterns

## Research Applications

This implementation is particularly useful for:

- **Behavioral Data Analysis**: Understanding sequential patterns in questionnaire responses
- **Feature Interaction Discovery**: Identifying which features influence each other
- **Causal Inference**: Testing statistical significance of learned patterns
- **Time Series Analysis**: Modeling dependencies in sequential data

## Dependencies

Key dependencies include:
- PyTorch 2.4.1
- NumPy 2.0.2
- Pandas 2.2.2
- Matplotlib & Seaborn (visualization)
- TensorBoard (logging)
- Scikit-learn (baseline models)

## License

[Add your license information here]

## Citation

[Add citation information if this is for a research paper]

## Contributing

[Add contribution guidelines if applicable]




Can you update it based on things you know about it?