import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
 
def demographic_parity_diff(y_pred, sensitive):
    groups = np.unique(sensitive)
    rates = {g: y_pred[sensitive==g].mean() for g in groups}
    return max(rates.values()) - min(rates.values()), rates
 
def equalised_odds_diff(y_true, y_pred, sensitive):
    groups = np.unique(sensitive)
    tpr = {}; fpr = {}
    for g in groups:
        mask = sensitive == g
        yt, yp = y_true[mask], y_pred[mask]
        tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        tpr[g] = tp/(tp+fn+1e-9)
        fpr[g] = fp/(fp+tn+1e-9)
    return max(tpr.values())-min(tpr.values()), max(fpr.values())-min(fpr.values())
 
def counterfactual_fairness_score(model, X, sensitive_col, n_samples=100):
    flipped = X.copy(); flipped[:, sensitive_col] = 1 - flipped[:, sensitive_col]
    orig = model.predict(X); flip = model.predict(flipped)
    return (orig == flip).mean()
 
np.random.seed(42)
n = 500
X = np.column_stack([
    np.random.randn(n),          # skill score
    np.random.randn(n),          # experience
    np.random.randint(0,2,n),    # gender (sensitive)
    np.random.randn(n)           # education
])
sensitive = X[:, 2].astype(int)
# Biased labels: female candidates less likely hired
y = ((X[:,0]+X[:,1]+X[:,3] + 0.8*X[:,2] + np.random.randn(n)*0.5) > 1).astype(int)
 
model = LogisticRegression(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
 
dpd, rates = demographic_parity_diff(y_pred, sensitive)
tpr_diff, fpr_diff = equalised_odds_diff(y, y_pred, sensitive)
cf = counterfactual_fairness_score(model, X, sensitive_col=2)
 
print(f"Demographic Parity Diff: {dpd:.3f}  (0=fair, rates={rates})")
print(f"Equal Opportunity Diff (TPR): {tpr_diff:.3f}")
print(f"Equal FPR Diff: {fpr_diff:.3f}")
print(f"Counterfactual Fairness: {cf:.3f}  (1.0=perfectly fair)")
