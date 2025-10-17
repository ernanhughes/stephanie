# stephanie/tools/export_goal_relevance_dataset.py
"""
Export goal-conditioned relevance dataset from evaluations for training MRQ/SICQL models.

This script creates a massive dataset of (goal, candidate_text, scores) triples that
encode how relevant documents/sections were judged with respect to specific goals.

Usage:
    python -m stephanie.tools.export_goal_relevance_dataset \
        --output-path=data/training/goal_relevance_dataset.csv \
        --dimensions=knowledge,clarity,grounding,overall \
        --min-score=0.0 \
        --limit=100000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from stephanie.models.base import engine
from stephanie.utils.file_utils import atomic_write

_logger = logging.getLogger(__name__)

def export_goal_relevance_dataset(
    output_path: str,
    dimensions: List[str] = None,
    min_score: float = 0.0,
    limit: Optional[int] = None,
    include_attributes: bool = True,
    format: str = "csv"  # "csv" or "jsonl"
) -> Dict[str, Any]:
    """
    Export goal-relevance dataset from document_evaluation_export_view.
    
    Args:
        output_path: Path to save exported dataset
        dimensions: Which score dimensions to include (None = all)
        min_score: Minimum score to include (filters noise)
        limit: Maximum rows to export (None = all)
        include_attributes: Include auxiliary attributes
        format: Output format ("csv" or "jsonl")
        
    Returns:
        Export statistics
    """
    if dimensions is None:
        dimensions = ["knowledge", "clarity", "grounding", "overall"]
    
    _logger.debug(f"Exporting goal-relevance dataset to {output_path}")
    _logger.debug(f"Dimensions: {dimensions}, Min score: {min_score}, Limit: {limit}")
    
    # Build query
    query = text("""
        SELECT 
            e.evaluation_id,
            e.goal_text,
            COALESCE(e.document_text, e.section_text) AS candidate_text,
            e.scores,
            e.attributes,
            e.created_at
        FROM document_evaluation_export_view e
        WHERE e.goal_text IS NOT NULL 
          AND (e.document_text IS NOT NULL OR e.section_text IS NOT NULL)
          AND LENGTH(COALESCE(e.document_text, e.section_text)) > 50
        ORDER BY e.created_at DESC
    """)
    
    if limit:
        query = text(f"{str(query)} LIMIT {limit}")
    
    # Execute query
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    
    _logger.debug(f"Fetched {len(rows)} raw evaluation records")
    
    # Process rows into training examples
    examples = []
    skipped = 0
    
    for row in rows:
        # Parse scores JSON
        try:
            scores = json.loads(row.scores) if isinstance(row.scores, str) else (row.scores or [])
        except Exception as e:
            _logger.warning(f"Failed to parse scores for evaluation {row.evaluation_id}: {str(e)}")
            skipped += 1
            continue
        
        # Parse attributes JSON
        attributes = {}
        if include_attributes:
            try:
                attributes = json.loads(row.attributes) if isinstance(row.attributes, str) else (row.attributes or {})
            except Exception:
                pass
        
        # Extract candidate text
        candidate_text = (row.document_text or row.section_text or "").strip()
        if not candidate_text or len(candidate_text) < 50:
            skipped += 1
            continue
            
        # Extract goal text
        goal_text = (row.goal_text or "").strip()
        if not goal_text:
            skipped += 1
            continue
        
        # Process each score dimension
        for score_record in scores:
            dimension = score_record.get("dimension", "")
            if dimension not in dimensions:
                continue
                
            score = score_record.get("score")
            if score is None:
                continue
                
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
                
            if score < min_score:
                continue
            
            # Create training example
            example = {
                "evaluation_id": row.evaluation_id,
                "goal_text": goal_text[:1000],  # Cap length for consistency
                "candidate_text": candidate_text[:5000],  # Cap length for consistency
                "dimension": dimension,
                "score": round(score, 4),
                "weight": float(score_record.get("weight", 1.0)),
                "source": score_record.get("source", "unknown"),
                "rationale": (score_record.get("rationale") or "")[:500],
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            
            # Add attributes if requested
            if include_attributes:
                # Flatten attributes for CSV compatibility
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        example[f"attr_{key}"] = value
                    elif isinstance(value, dict):
                        # JSON serialize nested dicts
                        example[f"attr_{key}"] = json.dumps(value, ensure_ascii=False)
                    else:
                        example[f"attr_{key}"] = str(value)
            
            examples.append(example)
    
    _logger.debug(f"Processed {len(examples)} training examples ({skipped} skipped)")
    
    # Save dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if format.lower() == "csv":
        _save_as_csv(examples, output_path)
    elif format.lower() == "jsonl":
        _save_as_jsonl(examples, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Generate statistics
    stats = _generate_statistics(examples, dimensions)
    stats["output_path"] = output_path
    stats["format"] = format
    stats["dimensions_exported"] = dimensions
    stats["min_score_threshold"] = min_score
    stats["rows_skipped"] = skipped
    stats["export_timestamp"] = datetime.now().isoformat()
    
    _logger.debug(f"Export complete. Statistics: {json.dumps(stats, indent=2)}")
    
    return stats

def _save_as_csv(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Save examples as CSV file"""
    if not examples:
        _logger.warning("No examples to save")
        return
    
    # Get all possible columns
    all_columns = set()
    for example in examples:
        all_columns.update(example.keys())
    
    # Sort columns for consistency
    columns = sorted(list(all_columns))
    
    with atomic_write(output_path, mode="w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for example in examples:
            # Ensure all columns are present
            row = {col: example.get(col, "") for col in columns}
            writer.writerow(row)
    
    _logger.debug(f"Saved {len(examples)} examples to {output_path}")

def _save_as_jsonl(examples: List[Dict[str, Any]], output_path: str) -> None:
    """Save examples as JSONL file"""
    with atomic_write(output_path, mode="w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    _logger.debug(f"Saved {len(examples)} examples to {output_path}")

def _generate_statistics(examples: List[Dict[str, Any]], dimensions: List[str]) -> Dict[str, Any]:
    """Generate dataset statistics"""
    if not examples:
        return {"total_examples": 0}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(examples)
    
    stats = {
        "total_examples": len(examples),
        "unique_goals": df["goal_text"].nunique(),
        "unique_candidates": df["candidate_text"].nunique(),
        "dimensions_distribution": {},
        "score_distribution": {},
        "source_distribution": df["source"].value_counts().to_dict() if "source" in df.columns else {},
    }
    
    # Dimension distribution
    for dim in dimensions:
        count = len(df[df["dimension"] == dim])
        stats["dimensions_distribution"][dim] = count
    
    # Score distribution per dimension
    for dim in dimensions:
        dim_df = df[df["dimension"] == dim]
        if not dim_df.empty:
            stats["score_distribution"][dim] = {
                "mean": round(dim_df["score"].mean(), 4),
                "std": round(dim_df["score"].std(), 4),
                "min": round(dim_df["score"].min(), 4),
                "max": round(dim_df["score"].max(), 4),
                "median": round(dim_df["score"].median(), 4),
            }
    
    # Length statistics
    stats["goal_text_length"] = {
        "mean": round(df["goal_text"].str.len().mean(), 1),
        "std": round(df["goal_text"].str.len().std(), 1),
        "min": int(df["goal_text"].str.len().min()),
        "max": int(df["goal_text"].str.len().max()),
    }
    
    stats["candidate_text_length"] = {
        "mean": round(df["candidate_text"].str.len().mean(), 1),
        "std": round(df["candidate_text"].str.len().std(), 1),
        "min": int(df["candidate_text"].str.len().min()),
        "max": int(df["candidate_text"].str.len().max()),
    }
    
    return stats

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Export goal-relevance dataset for training")
    parser.add_argument("--output-path", required=True, help="Path to save exported dataset")
    parser.add_argument("--dimensions", default="knowledge,clarity,grounding,overall", 
                       help="Comma-separated score dimensions to include")
    parser.add_argument("--min-score", type=float, default=0.0, 
                       help="Minimum score to include (filters noise)")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Maximum rows to export (None = all)")
    parser.add_argument("--format", default="csv", choices=["csv", "jsonl"],
                       help="Output format")
    parser.add_argument("--no-attributes", action="store_true",
                       help="Exclude auxiliary attributes")
    
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = [d.strip() for d in args.dimensions.split(",")] if args.dimensions else None
    
    # Run export
    try:
        stats = export_goal_relevance_dataset(
            output_path=args.output_path,
            dimensions=dimensions,
            min_score=args.min_score,
            limit=args.limit,
            include_attributes=not args.no_attributes,
            format=args.format
        )
        
        print(json.dumps(stats, indent=2))
        return 0
    except Exception as e:
        _logger.error(f"Export failed: {str(e)}", exc_info=True)
        return 1

# python -m stephanie.tools.export_goal_relevance_dataset --output-path=data/goal_relevance_dataset.csv --dimensions=knowledge,clarity,relevance --min-score=0.0 --limit=100
if __name__ == "__main__":
    sys.exit(main())