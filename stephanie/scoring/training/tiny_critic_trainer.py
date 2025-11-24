# stephanie/scoring/training/tiny_critic_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from stephanie.scoring.metrics.dynamic_features import load_core_metric_names

log = logging.getLogger(__name__)

DATA_PATH = Path("data/tiny_visicalc_critic.npz")
MODEL_PATH = Path("models/tiny_visicalc_critic.joblib")
VISUALIZATIONS_DIR = Path("data/visualizations/tiny_critic")
CORE_METRIC_PATH = Path("config/core_metrics.json")

# First 8 are the structural VisiCalc features (always leading dims in X)
CORE_FEATURE_NAMES = [
    "stability",
    "middle_dip",
    "std_dev",
    "sparsity",
    "entropy",
    "trend",
    "mid_bad_ratio",
    "frontier_util",
]


# ---------- Data & feature names ----------

def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset with detailed logging."""
    log.info("📂 Loading dataset from: %s", path.absolute())

    if not path.exists():
        log.error("❌ Dataset file not found: %s", path)
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        data = np.load(path)
        X = data["X"]
        y = data["y"]

        log.info(
            "✅ Dataset loaded successfully: "
            "X.shape=%s, y.shape=%s, dtypes: X=%s, y=%s",
            X.shape,
            y.shape,
            X.dtype,
            y.dtype,
        )

        # Log dataset statistics
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        log.info("📊 Class distribution: %s", class_distribution)
        log.info(
            "📈 Feature statistics - Min: %.3f, Max: %.3f, Mean: %.3f, Std: %.3f",
            float(X.min()),
            float(X.max()),
            float(X.mean()),
            float(X.std()),
        )

        # Ensure visualization directory exists
        VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

        return X, y

    except Exception as e:
        log.error("❌ Failed to load dataset %s: %s", path, e)
        raise


def build_feature_names(n_features: int) -> List[str]:
    """
    Build the list of feature names for the full X:

        [8 core visicalc names] + [dynamic metric names from core_metrics.json]

    If there is a mismatch with n_features, we truncate or pad with generic names.
    """
    # Dynamic metrics from MARS core summary
    dynamic_names = load_core_metric_names(CORE_METRIC_PATH)

    all_names = list(CORE_FEATURE_NAMES) + list(dynamic_names)
    total_defined = len(all_names)

    if total_defined < n_features:
        # Need to invent names for leftover dims (shouldn't normally happen)
        missing = n_features - total_defined
        log.warning(
            "⚠️  Only %d feature names defined for %d columns; "
            "inventing %d generic names",
            total_defined,
            n_features,
            missing,
        )
        for i in range(missing):
            all_names.append(f"feature_{total_defined + i}")

    elif total_defined > n_features:
        # More names than columns → likely fewer metrics in dataset than config
        log.warning(
            "⚠️  Truncating feature name list (%d → %d) to match dataset columns",
            total_defined,
            n_features,
        )
        all_names = all_names[:n_features]

    n_core = min(len(CORE_FEATURE_NAMES), n_features)
    n_dyn = max(0, n_features - n_core)
    log.info(
        "📌 Using %d total feature names (%d core + %d dynamic)",
        n_features,
        n_core,
        n_dyn,
    )
    return all_names[:n_features]


# ---------- Visualization helpers ----------

def safe_histplot(ax, x, y, bins=20):
    """
    Try KDE only if the feature looks continuous enough.
    Fall back to plain histogram when KDE would be singular.
    """
    # Quick heuristics to determine if KDE is feasible
    n_unique = np.unique(x).size
    std = float(np.std(x))
    try_kde = (n_unique >= 5) and (std > 1e-8) and (len(x) >= 10)
    
    try:
        # Try with KDE if conditions are met
        sns.histplot(x=x, hue=y, kde=try_kde, bins=bins, alpha=0.6, ax=ax)
    except Exception as e:
        log.warning("⚠️  KDE failed (%s). Falling back to kde=False.", e)
        # Fall back to histogram without KDE
        sns.histplot(x=x, hue=y, kde=False, bins=bins, alpha=0.6, ax=ax)

def visualize_features(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
    """
    Visualize distributions/correlations for the first 8 VisiCalc features.
    Uses safe_histplot to handle low-variance features that would break KDE.
    """
    log.info("📊 Creating feature visualizations (core VisiCalc features)...")

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Core feature histograms
    plt.figure(figsize=(15, 12))
    n_core = min(len(CORE_FEATURE_NAMES), X.shape[1], 8)

    for i in range(n_core):
        name = CORE_FEATURE_NAMES[i]
        plt.subplot(3, 3, i + 1)
        
        # Use the safe version that handles KDE failures
        safe_histplot(
            ax=plt.gca(),
            x=X[:, i],
            y=y,
            bins=15
        )
        
        plt.title(f"Distribution: {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / "feature_core_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("✅ Saved core feature histograms")

    # Correlation heatmap for the same core features
    import pandas as pd

    core_cols = min(n_core, X.shape[1])
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(
        X[:, :core_cols],
        columns=CORE_FEATURE_NAMES[:core_cols],
    )
    df["is_good"] = y
    corr = df.corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=corr.columns,
        yticklabels=corr.index,
    )
    plt.title("Feature Correlations (core VisiCalc features)")
    plt.tight_layout()
    plt.savefig(
        VISUALIZATIONS_DIR / "feature_core_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    log.info("✅ Saved core feature correlation heatmap")

    # Quick importance potential on core features only
    try:
        log.info("🔍 Core feature importance potential (|mean_good - mean_bad|):")
        for i in range(core_cols):
            diff = float(np.mean(X[y == 1, i]) - np.mean(X[y == 0, i]))
            log.info("   - %-20s %.4f", CORE_FEATURE_NAMES[i], abs(diff))
    except Exception as e:
        log.warning("⚠️  Could not calculate core feature importance potential: %s", e)


def evaluate_model_with_cv(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate with cross-validation and detailed metrics."""
    log.info("🔍 Starting cross-validation evaluation...")

    # Determine folds based on minority class count
    class_counts = np.bincount(y)
    n_splits = min(5, int(class_counts[class_counts > 0].min()))
    n_splits = max(n_splits, 2)  # at least 2 folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    log.info(
        "⚙️  Using %d folds for cross-validation (class_counts=%s)",
        n_splits,
        class_counts.tolist(),
    )

    accuracy_scores = []
    auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        log.info(
            "  ➡️ Fold %d/%d: Train=%d, Val=%d",
            fold + 1,
            n_splits,
            len(X_train),
            len(X_val),
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_proba))

        report = classification_report(
            y_val, y_pred, output_dict=True, zero_division=0
        )
        if "1" in report:
            precision_scores.append(report["1"]["precision"])
            recall_scores.append(report["1"]["recall"])
            f1_scores.append(report["1"]["f1-score"])

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Cross-Validation)")
    plt.savefig(
        VISUALIZATIONS_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    log.info("✅ Saved cross-validated confusion matrix")

    # ROC curve on full dataset after final fit
    try:
        plt.figure(figsize=(8, 6))
        model.fit(X, y)
        y_proba_full = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba_full)
        mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {mean_auc:.2f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Cross-Validation)")
        plt.legend(loc="lower right")
        plt.savefig(
            VISUALIZATIONS_DIR / "roc_curve.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        log.info("✅ Saved cross-validated ROC curve")
    except Exception as e:
        log.warning("⚠️  Could not generate ROC curve: %s", e)

    return {
        "accuracy_mean": float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
        "accuracy_std": float(np.std(accuracy_scores)) if len(accuracy_scores) > 1 else 0.0,
        "auc_mean": float(np.mean(auc_scores)) if auc_scores else 0.0,
        "auc_std": float(np.std(auc_scores)) if len(auc_scores) > 1 else 0.0,
        "precision_mean": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall_mean": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "f1_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "n_splits": n_splits,
    }


# ---------- Training ----------

def train_tiny_critic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[Any, Dict[str, Any], float, float]:
    """Train the tiny critic model with comprehensive logging."""
    log.info("🚀 Starting Tiny Critic training...")
    log.info("📦 Training dataset: %d samples, %d features", X.shape[0], X.shape[1])

    # Split data
    log.info("🔀 Splitting dataset into train/validation sets...")
    n_classes = len(np.unique(y))

    if X.shape[0] < 2 * n_classes:
        log.warning(
            "⚠️  Extremely small dataset (%d samples, %d classes) - "
            "using all data for training with minimal validation",
            X.shape[0],
            n_classes,
        )
        X_train, X_val = X, X[:1]
        y_train, y_val = y, y[:1]
    else:
        min_test_size = n_classes / X.shape[0]
        test_size = max(0.2, min_test_size)
        test_size = min(test_size, 0.3)

        log.info("   Using test_size=%.2f (min required: %.2f)", test_size, min_test_size)

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            log.warning("⚠️  Stratified split failed - using non-stratified split")
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
            )

    log.info(
        "📊 Split completed: Train=%d samples, Validation=%d samples",
        X_train.shape[0],
        X_val.shape[0],
    )

    # Model pipeline
    log.info("🔧 Building model pipeline (StandardScaler + LogisticRegression)...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    )

    log.info("🎯 Starting model training...")
    model.fit(X_train, y_train)
    log.info("✅ Model training completed successfully!")

    # Cross-validation
    cv_results = evaluate_model_with_cv(model, X, y)

    # Validation metrics
    log.info("🔮 Making predictions on validation set...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = float(accuracy_score(y_val, y_pred))
    auc = float(roc_auc_score(y_val, y_proba)) if len(np.unique(y_val)) > 1 else 0.0

    log.info("📊 Validation Results: Accuracy = %.4f, AUC-ROC = %.4f", acc, auc)
    log.info(
        "📈 Validation Details: Positive=%d, Negative=%d, Correct=%d/%d",
        int(np.sum(y_val == 1)),
        int(np.sum(y_val == 0)),
        int(np.sum(y_pred == y_val)),
        len(y_val),
    )

    # Save model
    log.info("💾 Saving model to: %s", MODEL_PATH.absolute())
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, MODEL_PATH)
        model_size = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0
        log.info(
            "✅ Model saved successfully! Size: %.2f KB, Path: %s",
            model_size / 1024,
            MODEL_PATH,
        )
    except Exception as e:
        log.error("❌ Failed to save model: %s", e)
        raise

    # Feature importance over ALL features (core + dynamic)
    try:
        logistic_model = model.named_steps["logisticregression"]
        coef = logistic_model.coef_[0]
        feature_importance = np.abs(coef)

        # Sort by importance
        idx_sorted = np.argsort(feature_importance)[::-1]
        log.info("🔍 Model Insights (top 20 features by |coef|):")
        top_k = min(len(feature_names), 20)
        for rank in range(top_k):
            idx = idx_sorted[rank]
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            log.info(
                "   %2d. %-30s %.4f",
                rank + 1,
                name,
                feature_importance[idx],
            )

        # Barplot for top 20
        plt.figure(figsize=(12, 6))
        top_idx = idx_sorted[:top_k]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = feature_importance[top_idx]
        sns.barplot(x=top_names, y=top_vals)
        plt.xticks(rotation=45, ha="right")
        plt.title("Top Feature Importances (|coef|)")
        plt.tight_layout()
        plt.savefig(
            VISUALIZATIONS_DIR / "feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        log.info("✅ Saved feature importance barplot")

    except Exception as e:
        log.warning("⚠️  Could not extract model insights: %s", e)

    return model, cv_results, acc, auc


# ---------- CLI entrypoint ----------

def main():
    """Main function with comprehensive logging setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("🎬 Starting Tiny Critic Trainer...")
    log.info("📁 Data path:  %s", DATA_PATH.absolute())
    log.info("📁 Model path: %s", MODEL_PATH.absolute())

    try:
        X, y = load_dataset(DATA_PATH)

        # Derive feature names from core + dynamic metrics
        feature_names = build_feature_names(X.shape[1])

        # Visualize core features
        visualize_features(X, y, feature_names)

        # Train model
        model, cv_results, accuracy, auc = train_tiny_critic(X, y, feature_names)

        # Cross-validation summary
        log.info(
            "📊 Cross-Validation Results (%d folds):", cv_results["n_splits"]
        )
        log.info(
            "   Accuracy:   %.4f ± %.4f",
            cv_results["accuracy_mean"],
            cv_results["accuracy_std"],
        )
        log.info(
            "   AUC-ROC:    %.4f ± %.4f",
            cv_results["auc_mean"],
            cv_results["auc_std"],
        )
        log.info("   Precision:  %.4f", cv_results["precision_mean"])
        log.info("   Recall:     %.4f", cv_results["recall_mean"])
        log.info("   F1-Score:   %.4f", cv_results["f1_mean"])

        # Final assessment
        log.info(
            "🎉 Tiny Critic training completed successfully!"
        )
        log.info(
            "🏆 Final Validation Results: Accuracy=%.4f, AUC-ROC=%.4f",
            accuracy,
            auc,
        )

        if cv_results["auc_mean"] > 0.9:
            performance_msg = "Excellent discrimination"
        elif cv_results["auc_mean"] > 0.8:
            performance_msg = "Good discrimination"
        elif cv_results["auc_mean"] > 0.7:
            performance_msg = "Fair discrimination"
        else:
            performance_msg = (
                "Poor discrimination – may need more data or feature engineering"
            )

        log.info("📋 Performance Assessment: %s", performance_msg)

        if X.shape[0] < 50:
            log.warning(
                "💡 Recommendation: Collect more data (aim for 100+ samples) "
                "for more reliable results (current: %d samples)",
                X.shape[0],
            )
        elif X.shape[0] < 100:
            log.warning(
                "💡 Recommendation: More data would improve reliability "
                "(current: %d samples)",
                X.shape[0],
            )
        else:
            log.info("✅ Sufficient data for reliable model training")

    except Exception as e:
        log.error("💥 Tiny Critic training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
