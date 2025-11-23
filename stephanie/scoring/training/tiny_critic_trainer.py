# stephanie/scoring/tiny_critic_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter

log = logging.getLogger(__name__)

DATA_PATH = Path("data/tiny_visicalc_critic.npz")
MODEL_PATH = Path("models/tiny_visicalc_critic.joblib")
VISUALIZATIONS_DIR = Path("data/visualizations/tiny_critic")
FEATURE_NAMES = [
    'stability', 'middle_dip', 'std_dev', 'sparsity',
    'entropy', 'trend', 'mid_bad_ratio', 'frontier_util'
]

def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset with detailed logging"""
    log.info(f"📂 Loading dataset from: {path.absolute()}")
    
    if not path.exists():
        log.error(f"❌ Dataset file not found: {path}")
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    try:
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        
        log.info(f"✅ Dataset loaded successfully: "
                f"X.shape={X.shape}, y.shape={y.shape}, "
                f"dtypes: X={X.dtype}, y={y.dtype}")
        
        # Log dataset statistics
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        log.info(f"📊 Class distribution: {class_distribution}")
        log.info(f"📈 Feature statistics - "
                f"Min: {X.min():.3f}, Max: {X.max():.3f}, "
                f"Mean: {X.mean():.3f}, Std: {X.std():.3f}")
        
        # Create visualizations directory
        VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        return X, y
        
    except Exception as e:
        log.error(f"❌ Failed to load dataset {path}: {e}")
        raise

def visualize_features(X: np.ndarray, y: np.ndarray):
    """Create visualizations of feature distributions by class"""
    log.info("📊 Creating feature visualizations...")
    
    # Ensure directory exists
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    plt.figure(figsize=(15, 12))
    
    # 1. Distribution plots for each feature
    for i, name in enumerate(FEATURE_NAMES):
        plt.subplot(3, 3, i+1)
        sns.histplot(data={'Good': X[y == 1, i], 'Bad': X[y == 0, i]}, 
                    kde=True, bins=15, alpha=0.6)
        plt.title(f'Distribution: {name}')
        plt.xlabel('Value')
        plt.ylabel('Count')
    
    # 2. Box plots to show differences between classes
    plt.subplot(3, 3, 8)
    box_data = []
    for i, name in enumerate(FEATURE_NAMES):
        for cls, label in enumerate(['Bad', 'Good']):
            box_data.append({
                'Feature': name,
                'Value': X[y == cls, i],
                'Class': label
            })
    
    # Convert to format for seaborn
    all_values = []
    all_features = []
    all_classes = []
    for item in box_data:
        all_values.extend(item['Value'])
        all_features.extend([item['Feature']] * len(item['Value']))
        all_classes.extend([item['Class']] * len(item['Value']))
    
    sns.boxplot(x=all_features, y=all_values, hue=all_classes)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Distributions by Class')
    plt.tight_layout()
    
    # 3. Correlation heatmap
    plt.subplot(3, 3, 9)
    plt.title('Feature Correlations')
    # Add target variable for correlation with class
    X_with_target = np.column_stack([X, y])
    feature_names = FEATURE_NAMES + ['is_good']
    corr = np.corrcoef(X_with_target, rowvar=False)
    sns.heatmap(corr, annot=True, cmap='coolwarm', 
                xticklabels=feature_names, 
                yticklabels=feature_names, fmt='.2f')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(VISUALIZATIONS_DIR / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✅ Feature visualizations saved to {VISUALIZATIONS_DIR}")
    
    # 4. Feature importance potential analysis
    try:
        # Simple analysis: mean difference between classes
        mean_diffs = []
        for i in range(X.shape[1]):
            diff = np.mean(X[y == 1, i]) - np.mean(X[y == 0, i])
            mean_diffs.append((FEATURE_NAMES[i], abs(diff)))
        
        # Sort by absolute difference
        mean_diffs.sort(key=lambda x: x[1], reverse=True)
        
        log.info("🔍 Feature importance potential (based on mean differences):")
        for name, diff in mean_diffs:
            log.info(f"   - {name}: {diff:.4f}")
    except Exception as e:
        log.warning(f"⚠️  Could not calculate feature importance potential: {e}")

def evaluate_model_with_cv(model, X, y):
    """Evaluate with cross-validation and detailed metrics"""
    log.info("🔍 Starting cross-validation evaluation...")
    
    cv = StratifiedKFold(n_splits=min(5, sum(y > 0)), 
                         shuffle=True, 
                         random_state=42)
    
    # For very small datasets, reduce folds
    n_splits = cv.get_n_splits()
    log.info(f"⚙️  Using {n_splits} folds for cross-validation (adjusted for small dataset)")
    
    # Store all metrics
    accuracy_scores = []
    auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # For confusion matrix analysis
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        log.info(f"  ➡️ Fold {fold+1}/{n_splits}: "
                f"Train={len(X_train)}, Val={len(X_val)}")
        
        # Train model on this fold
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Store for overall metrics
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_proba))
        
        # For binary classification with small samples
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        if '1' in report:  # Class 1 is "good" reasoning
            precision_scores.append(report['1']['precision'])
            recall_scores.append(report['1']['recall'])
            f1_scores.append(report['1']['f1-score'])
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Cross-Validation)')
    plt.savefig(VISUALIZATIONS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ROC curve
    try:
        from sklearn.metrics import roc_curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(all_y_true, [p[1] for p in model.predict_proba(X)])
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {np.mean(auc_scores):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Cross-Validation)')
        plt.legend(loc="lower right")
        plt.savefig(VISUALIZATIONS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.warning(f"⚠️  Could not generate ROC curve: {e}")
    
    return {
        'accuracy_mean': np.mean(accuracy_scores) if accuracy_scores else 0.0,
        'accuracy_std': np.std(accuracy_scores) if len(accuracy_scores) > 1 else 0.0,
        'auc_mean': np.mean(auc_scores) if auc_scores else 0.0,
        'auc_std': np.std(auc_scores) if len(auc_scores) > 1 else 0.0,
        'precision_mean': np.mean(precision_scores) if precision_scores else 0.0,
        'recall_mean': np.mean(recall_scores) if recall_scores else 0.0,
        'f1_mean': np.mean(f1_scores) if f1_scores else 0.0,
        'n_splits': n_splits
    }

def train_tiny_critic(X: np.ndarray, y: np.ndarray):
    """Train the tiny critic model with comprehensive logging"""
    log.info("🚀 Starting Tiny Critic training...")
    
    # Log initial dataset info
    log.info(f"📦 Training dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split the data - robust version for small datasets
    log.info("🔀 Splitting dataset into train/validation sets...")
    n_classes = len(np.unique(y))

    # For very small datasets, we need to adjust test_size
    if X.shape[0] < 2 * n_classes:
        log.warning(f"⚠️  Extremely small dataset ({X.shape[0]} samples, {n_classes} classes) - using all data for training")
        X_train, X_val = X, X[:1]  # Create minimal validation set
        y_train, y_val = y, y[:1]
    else:
        # Calculate minimum test size to have at least 1 sample per class
        min_test_size = n_classes / X.shape[0]
        # Use at least 20% for test if possible, but ensure enough samples per class
        test_size = max(0.2, min_test_size)
        # Cap at 30% to ensure enough training data
        test_size = min(test_size, 0.3)
        
        log.info(f"   Using test_size={test_size:.2f} (min required: {min_test_size:.2f})")
        
        try:
            # Try stratified split first
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # Fall back to non-stratified if stratified fails
            log.warning("⚠️  Stratified split failed - using non-stratified split")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

    log.info(f"📊 Split completed: "
            f"Train: {X_train.shape[0]} samples, "
            f"Validation: {X_val.shape[0]} samples")
    
    # Create model pipeline
    log.info("🔧 Building model pipeline (StandardScaler + LogisticRegression)...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,  # Increased for small datasets
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        ),
    )
    
    # Train the model
    log.info("🎯 Starting model training...")
    model.fit(X_train, y_train)
    log.info("✅ Model training completed successfully!")
    
    # Evaluate with cross-validation (more reliable for small datasets)
    cv_results = evaluate_model_with_cv(model, X, y)
    
    # Make predictions on validation set
    log.info("🔮 Making predictions on validation set...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.0
    
    log.info(f"📊 Validation Results: "
            f"Accuracy = {acc:.4f}, "
            f"AUC-ROC = {auc:.4f}")
    
    # Log detailed performance breakdown
    val_positives = np.sum(y_val == 1)
    val_negatives = np.sum(y_val == 0)
    correct_predictions = np.sum(y_pred == y_val)
    
    log.info(f"📈 Validation Details: "
            f"Positive samples: {val_positives}, "
            f"Negative samples: {val_negatives}, "
            f"Correct predictions: {correct_predictions}/{len(y_val)}")
    
    # Save the model
    log.info(f"💾 Saving model to: {MODEL_PATH.absolute()}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Ensuring model directory exists: {MODEL_PATH.parent}")
    
    try:
        joblib.dump(model, MODEL_PATH)
        model_size = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0
        model_size_kb = model_size / 1024
        log.info(f"✅ Model saved successfully! "
                f"Size: {model_size_kb:.2f} KB, "
                f"Path: {MODEL_PATH}")
    except Exception as e:
        log.error(f"❌ Failed to save model: {e}")
        raise
    
    # Log model insights
    try:
        # Get feature importance from the logistic regression
        logistic_model = model.named_steps['logisticregression']
        feature_importance = np.abs(logistic_model.coef_[0])
        top_feature_idx = np.argmax(feature_importance)
        
        log.info(f"🔍 Model Insights: ")
        log.info(f"   Most important feature: {FEATURE_NAMES[top_feature_idx]}")
        log.info(f"   Importance: {feature_importance[top_feature_idx]:.4f}")
        
        # Log all feature importances
        log.info("   Feature importances:")
        for i, name in enumerate(FEATURE_NAMES):
            log.info(f"      - {name}: {feature_importance[i]:.4f}")
        
        # Save feature importance to visualization
        plt.figure(figsize=(10, 6))
        sns.barplot(x=FEATURE_NAMES, y=feature_importance)
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(VISUALIZATIONS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log.warning(f"⚠️  Could not extract model insights: {e}")
    
    return model, cv_results, acc, auc

def main():
    """Main function with comprehensive logging setup"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    log.info("🎬 Starting Tiny Critic Trainer...")
    log.info(f"📁 Data path: {DATA_PATH.absolute()}")
    log.info(f"📁 Model path: {MODEL_PATH.absolute()}")
    
    try:
        # Load dataset
        X, y = load_dataset(DATA_PATH)
        
        # Create feature visualizations
        visualize_features(X, y)
        
        # Train model
        model, cv_results, accuracy, auc = train_tiny_critic(X, y)
        
        # Log cross-validation results
        log.info(f"📊 Cross-Validation Results ({cv_results['n_splits']} folds):")
        log.info(f"   Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        log.info(f"   AUC-ROC: {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")
        log.info(f"   Precision: {cv_results['precision_mean']:.4f}")
        log.info(f"   Recall: {cv_results['recall_mean']:.4f}")
        log.info(f"   F1-Score: {cv_results['f1_mean']:.4f}")
        
        # Final success message
        log.info("🎉 Tiny Critic training completed successfully!")
        log.info(f"🏆 Final Results: Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}")
        
        # Performance interpretation
        if cv_results['auc_mean'] > 0.9:
            performance_msg = "Excellent discrimination"
        elif cv_results['auc_mean'] > 0.8:
            performance_msg = "Good discrimination" 
        elif cv_results['auc_mean'] > 0.7:
            performance_msg = "Fair discrimination"
        else:
            performance_msg = "Poor discrimination - may need more data or feature engineering"
            
        log.info(f"📋 Performance Assessment: {performance_msg}")
        
        # Data size recommendation
        if X.shape[0] < 50:
            log.warning("💡 Recommendation: Collect more data (aim for 100+ samples) for more reliable results")
        elif X.shape[0] < 100:
            log.warning("💡 Recommendation: More data would improve reliability (current: {} samples)".format(X.shape[0]))
        else:
            log.info("✅ Sufficient data for reliable model training")
            
    except Exception as e:
        log.error(f"💥 Tiny Critic training failed: {e}")
        raise

if __name__ == "__main__":
    main()