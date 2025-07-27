import matplotlib.pyplot as plt
import pandas as pd
import os

def save_dataframe_plot(df: pd.DataFrame, 
                        title: str, 
                        filename: str, 
                        x_label: str = None, 
                        y_label: str = None, 
                        rotation: int = 45, 
                        figsize=(12, 6), 
                        tight=True, 
                        output_dir="plots"):
    """
    Generate and save a bar chart from a DataFrame.

    Args:
        df (pd.DataFrame): The dataframe to plot (must have one row).
        title (str): Plot title.
        filename (str): Filename to save the image as (without directory).
        x_label (str): Optional label for x-axis.
        y_label (str): Optional label for y-axis.
        rotation (int): Rotation for x-tick labels.
        figsize (tuple): Size of the figure.
        tight (bool): Whether to call plt.tight_layout().
        output_dir (str): Directory to save plots.
    """
    if df.empty or len(df) != 1:
        raise ValueError("Expected a single-row DataFrame.")

    os.makedirs(output_dir, exist_ok=True)
    data = df.iloc[0]

    plt.figure(figsize=figsize)
    data.plot(kind='bar')
    plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    if tight:
        plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Plot saved to {path}")
