import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from pygam import LinearGAM, s, l

# 1. Load California housing data
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. JSON spline configuration
terms_json = """
[
  {"type": "spline", "scale": false, "knots": 12, "lambda": 2.0273110025405088},
  {"type": "spline", "scale": false, "knots": 7, "lambda": 1.2260178579008096},
  {"type": "none", "scale": false, "knots": null, "lambda": null},
  {"type": "spline", "scale": false, "knots": 12, "lambda": 9.626868308482564},
  {"type": "none", "scale": false, "knots": null, "lambda": null},
  {"type": "spline", "scale": true, "knots": 20, "lambda": 1.1753501900584824},
  {"type": "spline", "scale": true, "knots": 20, "lambda": 0.4881512042118739},
  {"type": "spline", "scale": true, "knots": 15, "lambda": 0.258562295600376}
]
"""
terms_data = json.loads(terms_json)

# 3. Build GAM terms and lambda array
gam_terms = []
lambdas = []
feature_names = list(X.columns)
n_features = X.shape[1]

for i, term in enumerate(terms_data):
    feat_idx = i % n_features  # wrap around if more terms than features
    if term["type"] == "spline":
        gam_terms.append(s(feat_idx, n_splines=term["knots"]))
        lambdas.append(term["lambda"])
    else:
        gam_terms.append(l(feat_idx))  # linear term
        lambdas.append(0.6)  # default lambda for linear terms

# 4. Combine terms using + operator
combined_terms = gam_terms[0]
for term in gam_terms[1:]:
    combined_terms += term

# 5. Fit GAM with combined terms
gam = LinearGAM(combined_terms, lam=lambdas).fit(X.values, y.values)

# 6. Extract EDF
edf_total = gam.statistics_['edof']

# 7. Heuristic EDF
total_splines = sum(term["knots"] for term in terms_data if term["type"] == "spline")
heuristic_edf = total_splines * 0.3

# 8. Print results
print("Feature names:", feature_names)
print("Total EDF (pyGAM):", edf_total)
print("Total splines (sum of knots):", total_splines)
print("Heuristic EDF (0.3 * total splines):", heuristic_edf)