# tools/visblog_eval.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def run_eval(
    output_dir: Path = Path("output/eval"),
    num_reviews: int = 3,
    reviewers: List[str] = ["Reviewer 1", "Reviewer 2", "Reviewer 3"],
) -> None:
    """Run evaluation of different outline types"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define outline types to compare
    outline_types = ["paper_only", "chat_only", "visblog"]
    
    # Initialize results structure
    results = {
        outline_type: {
            "coverage": [],
            "coherence": [],
            "novelty": []
        } for outline_type in outline_types
    }
    
    # Collect reviews from all reviewers
    for reviewer in reviewers:
        print(f"\n{'='*50}")
        print(f"Reviewing by {reviewer}")
        print(f"{'='*50}")
        
        for outline_type in outline_types:
            print(f"\nReviewing {outline_type} outline")
            outline_path = Path(f"output/{outline_type}_outline.md")
            if not outline_path.exists():
                print(f"Warning: {outline_path} does not exist")
                continue
            
            # Read the outline
            with open(outline_path, "r") as f:
                outline_text = f.read()
            
            print(f"\n{'='*50}")
            print(f"{outline_type} Outline:")
            print(f"{'='*50}")
            print(outline_text[:500] + "..." if len(outline_text) > 500 else outline_text)
            print(f"{'='*50}\n")
            
            # Collect scores
            coverage = float(input("Coverage (0-5): "))
            coherence = float(input("Coherence (0-5): "))
            novelty = float(input("Novelty (0-5): "))
            
            results[outline_type]["coverage"].append(coverage)
            results[outline_type]["coherence"].append(coherence)
            results[outline_type]["novelty"].append(novelty)
    
    # Save raw scores to CSV
    csv_path = output_dir / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Reviewer", "Outline Type", "Coverage", "Coherence", "Novelty"])
        
        for reviewer_idx, reviewer in enumerate(reviewers):
            for outline_type in outline_types:
                writer.writerow([
                    reviewer,
                    outline_type,
                    results[outline_type]["coverage"][reviewer_idx],
                    results[outline_type]["coherence"][reviewer_idx],
                    results[outline_type]["novelty"][reviewer_idx]
                ])
    
    print(f"\nRaw scores saved to {csv_path}")
    
    # Calculate average scores
    avg_results = {}
    for outline_type in outline_types:
        avg_results[outline_type] = {
            "coverage": np.mean(results[outline_type]["coverage"]),
            "coherence": np.mean(results[outline_type]["coherence"]),
            "novelty": np.mean(results[outline_type]["novelty"])
        }
    
    # Save average scores to JSON
    json_path = output_dir / "avg_results.json"
    with open(json_path, "w") as f:
        json.dump(avg_results, f, indent=2)
    
    print(f"Average scores saved to {json_path}")
    
    # Generate plots
    generate_plots(avg_results, output_dir, reviewers)
    
    # Print summary table
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    print(f"{'Outline Type':<15} {'Coverage':>8} {'Coherence':>10} {'Novelty':>8}")
    print("-"*50)
    for outline_type in outline_types:
        print(f"{outline_type:<15} {avg_results[outline_type]['coverage']:>8.2f} "
              f"{avg_results[outline_type]['coherence']:>10.2f} "
              f"{avg_results[outline_type]['novelty']:>8.2f}")
    print("="*50)


def generate_plots(avg_results: Dict[str, Dict[str, float]], output_dir: Path, reviewers: List[str]) -> None:
    """Generate and save comparison plots"""
    # Bar plot for average scores
    plt.figure(figsize=(10, 6))
    categories = ["Coverage", "Coherence", "Novelty"]
    outline_types = list(avg_results.keys())
    values = [[avg_results[ot][cat.lower()] for ot in outline_types] for cat in categories]
    
    x = np.arange(len(outline_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cat in enumerate(categories):
        ax.bar(x + i*width, values[i], width, label=cat)
    
    ax.set_ylabel("Score (0-5)")
    ax.set_title("Average Scores by Outline Type")
    ax.set_xticks(x + width)
    ax.set_xticklabels(outline_types)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "bar_plot.png")
    plt.close()
    
    # Radar chart for comparison
    categories = ["Coverage", "Coherence", "Novelty"]
    num_vars = len(categories)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Set up the radar chart
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # close the loop
    
    # Plot each outline type
    for outline_type in outline_types:
        values = [
            avg_results[outline_type]["coverage"],
            avg_results[outline_type]["coherence"],
            avg_results[outline_type]["novelty"]
        ]
        values += values[:1]  # close the loop
        
        ax.plot(angles, values, linewidth=2, label=outline_type)
        ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(0, 5)
    
    plt.title("Outline Comparison Radar Chart", size=15, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "radar_plot.png")
    plt.close()
    
    # Box plot for all reviews
    data = []
    labels = []
    for outline_type in outline_types:
        data.append(avg_results[outline_type]["coverage"])
        data.append(avg_results[outline_type]["coherence"])
        data.append(avg_results[outline_type]["novelty"])
        labels.extend([f"{outline_type}\nCoverage", f"{outline_type}\nCoherence", f"{outline_type}\nNovelty"])
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Score (0-5)")
    plt.title("Score Distribution by Outline Type and Metric")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "box_plot.png")
    plt.close()


if __name__ == "__main__":
    run_eval()