import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model fit
with open("out/microcredit.pkl", "rb") as f:
    fit = pickle.load(f)["mg"]

theta_df = fit["theta"]
lambda_df = fit["lambda"]

# Load original data (required for empirical means & PPC)
from pyreadr import read_r
result = read_r("microcredit-profit-independent.Rdata")
data = result["data"]
data = data.rename(columns={"site": "g", "treatment": "t", "profit": "y"})
data["g"] = data["g"].astype("category")
data["t"] = data["t"].astype("category")

# Posterior vs empirical mean plots (same as your existing code)
# [Insert your posterior_vs_empirical and PPC code here as-is]

# Save all plots to figs/
