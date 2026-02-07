import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from yellowbrick.classifier import ClassificationReport

def test_yellowbrick():
    try:
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier()
        
        # This is the line that triggers the dependency drift in 2026
        visualizer = ClassificationReport(model)
        visualizer.fit(X, y)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_yellowbrick()