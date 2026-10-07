from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score


def train_classical(X_train, y_train, X_val, y_val, c_grid, gamma_grid, seed=42):
    best = None
    score = -1
    for c in c_grid:
        for gamma in gamma_grid:
            model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel='rbf', random_state=seed))
            model.fit(X_train, y_train)
            candidate = f1_score(y_val, model.predict(X_val), zero_division=0)
            if candidate > score:
                best, score = model, candidate
    return best, {'C': best[-1].C, 'gamma': best[-1].gamma, 'validation_f1': float(score)}
