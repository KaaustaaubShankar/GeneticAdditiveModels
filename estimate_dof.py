import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from pygam import LinearGAM, s

def gam_feature_importance_by_feature(gam, X, feature_to_term):
    """
    Return importance array aligned with original feature order.
    importance[i] = variance of partial dependence for feature i if present,
    otherwise 0.0 for excluded features.
    """
    importance = np.zeros(X.shape[1], dtype=float)
    for feat_idx, term_idx in enumerate(feature_to_term):
        if term_idx is None:
            importance[feat_idx] = 0.0
            continue
        # get partial dependence for the term (use X grid = column values for that feature)
        # pygam.partial_dependence(term=term_idx, X=...) returns array of shape (n_points,)
        pd = gam.partial_dependence(term=term_idx, X=X.values)  # contribution for each row
        # if pd is 2D (sometimes returns (grid, values)), take the values column
        if isinstance(pd, tuple) or (hasattr(pd, "ndim") and pd.ndim > 1 and pd.shape[1] > 1):
            # handle possible return shapes robustly
            # prefer last column as contributions
            pd = np.asarray(pd)
            if pd.ndim == 2:
                pd_vals = pd[:, -1]
            else:
                pd_vals = pd.ravel()
        else:
            pd_vals = np.asarray(pd).ravel()
        importance[feat_idx] = float(np.var(pd_vals))
    return importance

# 1. Load California housing data
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. JSON spline configuration (your example)
terms_json = """
[
  {
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 8.267409055174769
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 8,
    "lambda": 4.044812906535987
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 0.9944962268218194
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 15,
    "lambda": 0.25603580044899665
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 12,
    "lambda": 7.613477482756874
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 18,
    "lambda": 3.6426630613116258
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 15,
    "lambda": 3.52883952014102
  }
]
"""
terms_data = json.loads(terms_json)

# 3. Build GAM terms and lambda array, track mapping from original feature -> term index
gam_terms = []
lambdas = []
feature_to_term = [None] * X.shape[1]  # mapping: feature_idx -> term_idx in GAM (or None if excluded)

feature_names = list(X.columns)
n_features = X.shape[1]

# terms_data length should correspond to features; if longer, we wrap or truncate as you need.
# Here we assume terms_data length == n_features (as in your example).
term_idx = 0
for i, term in enumerate(terms_data):
    feat_idx = i % n_features  # keep aligned to features
    if term["type"] == "spline":
        # create a spline term on this feature
        n_splines = max(4, int(term["knots"]))  # ensure sensible lower bound
        gam_terms.append(s(feat_idx, n_splines=n_splines))
        lambdas.append(float(term["lambda"]) if term["lambda"] is not None else 1.0)
        feature_to_term[feat_idx] = term_idx
        term_idx += 1
    elif term["type"] == "linear":
        # approximate linear with 2 splines + very large lambda
        gam_terms.append(s(feat_idx, n_splines=2))
        lambdas.append(1e6)  # very strong penalty -> effectively linear
        feature_to_term[feat_idx] = term_idx
        term_idx += 1
    else:  # "none" -> skip this feature (excluded)
        feature_to_term[feat_idx] = None
        # do not add a term or lambda

# If we ended up with zero terms (all 'none'), add a trivial intercept/small linear fallback
if len(gam_terms) == 0:
    # add tiny linear-like terms for each feature (fallback)
    for i in range(n_features):
        gam_terms.append(s(i, n_splines=2))
        lambdas.append(1e6)
        feature_to_term[i] = len(gam_terms) - 1

# 4. Combine terms using + operator
combined_terms = gam_terms[0]
for term in gam_terms[1:]:
    combined_terms += term

# 5. Fit GAM with combined terms
# Ensure lambdas array length equals number of terms or pass scalar
gam = LinearGAM(combined_terms, lam=lambdas).fit(X.values, y.values)

# 6. Extract EDF and importance
edf_total = gam.statistics_['edof']
importance_by_feature = gam_feature_importance_by_feature(gam, X, feature_to_term)

# 7. Heuristic EDF
total_splines = sum(int(term["knots"]) for term in terms_data if term["type"] == "spline")
heuristic_edf = total_splines * 0.3

# 8. Print results (neat)
print("Feature names:", feature_names)
print("Feature -> term index mapping:", feature_to_term)
print("\nImportance (variance of partial dependence) per feature:")
for name, imp in zip(feature_names, importance_by_feature):
    print(f"  {name:10s} : {imp:.6f}")

print("\nTotal EDF (pyGAM):", edf_total)
print("Total splines (sum of knots):", total_splines)
print("Heuristic EDF (0.3 * total splines):", heuristic_edf)
