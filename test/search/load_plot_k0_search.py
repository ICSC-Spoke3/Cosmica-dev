import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nevergrad as ng
def plot_best_x(parquet_path: str):

    df = pd.read_parquet(parquet_path)

    # Validate 'best_x' exists and is scalar
    if 'best_x' not in df.columns:
        raise ValueError("'best_x' column not found in the dataset.")

    if df['best_x'].apply(lambda x: isinstance(x, (int, float))).all() == False:
        raise ValueError("Some 'best_x' entries are not scalar numbers.")

    # Calculate mean and variance
    mean_best_x = df['best_x'].mean()
    var_best_x = df['best_x'].var()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['best_x'], bins=30, kde=True, color="skyblue")
    plt.axvline(mean_best_x, color='red', linestyle='--', label=f'Mean = {mean_best_x:.4f}')
    plt.title('Distribution of best_x')
    plt.xlabel('best_x')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    # Annotate with variance
    plt.text(0.95, 0.95, f'Variance = {var_best_x:.4f}',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.tight_layout()
    plt.show()

# Example usage
parquet_path = Path(__file__).parent / 'parquet' / 'k0_search'
plot_best_x(parquet_path)
