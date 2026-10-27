import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_export_model():
    """
    Trains a robust 1568-dimensional multimodal emergency triage Random Forest model
    aligned with the paper architecture (768 ViT + 768 DistilBERT + 32 IoT/Spatial).
    Exports the trained model to triage_app/models/trained_triage_rf.joblib.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "trained_triage_rf.joblib")
    
    print("Generating multimodal feature representations for training...")
    np.random.seed(42)
    n_samples = 3000
    
    # 4 Triage Classes:
    # 0: P1 - Critical (Severe fire, collapsed structures, multi-vehicle trauma, lethal gas/temp)
    # 1: P2 - High (Significant damage, active hazard, moderate casualties)
    # 2: P3 - Moderate (Localized incident, minor injuries, stable environment)
    # 3: P4 - Low (Property-only or false alarm, normal vitals/sensors)
    
    y = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])
    
    # 1. Visual Features (768-D, simulating ViT CLS tokens)
    # Visual cues correlate strongly with class severity
    vis_base = np.random.randn(n_samples, 768) * 0.5
    for i in range(4):
        mask = (y == i)
        # Shift distinct clusters in latent space
        vis_base[mask, :64] += (3 - i) * 1.2
        vis_base[mask, 64:128] += (i - 1.5) * 0.8
        
    # 2. Textual Features (768-D, simulating DistilBERT urgency embeddings)
    txt_base = np.random.randn(n_samples, 768) * 0.5
    for i in range(4):
        mask = (y == i)
        txt_base[mask, :64] += (3 - i) * 1.1
        txt_base[mask, 128:192] += (2.5 - i) * 0.9

    # 3. Spatio-Temporal & IoT Features (32-D)
    # Indices:
    # 0: Temperature Normalized (0 to 1)
    # 1: Smoke / Gas PPM (0 to 1)
    # 2: Proximity to nearest hospital (0 to 1, inverse)
    # 3: Structural vibration / accelerometer
    # 4..31: Contextual indicators
    iot_base = np.random.rand(n_samples, 32)
    for i in range(4):
        mask = (y == i)
        # High hazard sensors for P1 & P2
        if i == 0:  # P1 Critical
            iot_base[mask, 0] = np.random.uniform(0.75, 1.0, size=np.sum(mask)) # High temp
            iot_base[mask, 1] = np.random.uniform(0.80, 1.0, size=np.sum(mask)) # Dense smoke / lethal gas
            iot_base[mask, 3] = np.random.uniform(0.70, 1.0, size=np.sum(mask)) # High vibration/collapse
        elif i == 1: # P2 High
            iot_base[mask, 0] = np.random.uniform(0.50, 0.80, size=np.sum(mask))
            iot_base[mask, 1] = np.random.uniform(0.50, 0.75, size=np.sum(mask))
        elif i == 2: # P3 Moderate
            iot_base[mask, 0] = np.random.uniform(0.25, 0.55, size=np.sum(mask))
            iot_base[mask, 1] = np.random.uniform(0.20, 0.50, size=np.sum(mask))
        else: # P4 Low
            iot_base[mask, 0] = np.random.uniform(0.05, 0.30, size=np.sum(mask))
            iot_base[mask, 1] = np.random.uniform(0.00, 0.20, size=np.sum(mask))

    # Early Multimodal Fusion: (N, 768 + 768 + 32) = (N, 1568)
    X_fused = np.concatenate([vis_base, txt_base, iot_base], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("Training Balanced Random Forest Classifier (100 estimators, max_depth=15)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"Validation Accuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["P1-Critical", "P2-High", "P3-Moderate", "P4-Low"]))
    
    # Save the model
    joblib.dump(clf, model_path)
    print(f"Successfully exported model checkpoint to: {model_path}")
    return model_path

if __name__ == "__main__":
    train_and_export_model()
