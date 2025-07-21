import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

def main():
    y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
    sigma = np.sqrt(np.array([15, 10, 16, 11, 9, 11, 10, 18]))
    schools = [f"School {i+1}" for i in range(8)]

    def generate_synthetic_data(y,sigma,n_per_school):
        data=[]
        for i in range(len(y)):
            std = sigma[i]
            outcomes = np.random.normal(loc=y[i],scale=std,size=n_per_school)
            data.append(outcomes)
        return np.array(data)

    # def fit_pymc_model(y_obs, sigma_obs):
    def fit_pymc_model(synthetic):
        n_schools = synthetic.shape[0]
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=10)
            tau = pm.HalfCauchy("tau", beta=5)
            # tau = pm.HalfNormal("tau", sigma=0.3) # shrinks individual school effects, deliberate misspecification
            
            # theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(y_obs))
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_schools)
            
            # y_obs_ = pm.Normal("y_obs", mu=theta, sigma=sigma_obs, observed=y_obs)
            for j in range(n_schools):
                pm.Normal(f"y_obs_{j}",mu=theta[j],sigma=1.0,observed=synthetic[j])
                
            trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.95, return_inferencedata=True)
            return trace

    # def bayesbag(y_obs, sigma_obs, B=100, mfactor=1):
    def bayesbag(synthetic, B=100, mfactor=1):
        n_schools = synthetic.shape[0]
        bagged_thetas = []
        # N = len(y_obs)
        N = n_schools
        M = int(mfactor * N)
        for _ in tqdm(range(B), desc="BayesBag iterations"):
            idx = np.random.choice(N, size=M, replace=True)
            # y_boot = y_obs[idx]
            # sigma_boot = sigma_obs[idx]
            
            sampled_data = synthetic[idx]

            # trace = fit_pymc_model(y_boot, sigma_boot)
            trace = fit_pymc_model(sampled_data)
            mean_theta = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
            bagged_thetas.append(mean_theta)
        return np.array(bagged_thetas)

    # PREVIOUS WORKFLOW, NO DATA GENERATION
    # bagged_theta_samples = bayesbag(y, sigma, B=100)
    # theta_bagged_mean = bagged_theta_samples.mean(axis=0)
    # theta_bagged_std = bagged_theta_samples.std(axis=0)

    # trace = fit_pymc_model(y, sigma)
    # theta_standard_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
    # theta_standard_std = trace.posterior["theta"].std(dim=["chain", "draw"]).values

    # CURRENT WORKFLOW WITH DATA GENERATION
    synthetic_data = generate_synthetic_data(y,sigma,n_per_school=30)
    trace_standard = fit_pymc_model(synthetic_data)
    theta_standard_mean = trace_standard.posterior["theta"].mean(dim=["chain","draw"]).values
    theta_standard_std = trace_standard.posterior["theta"].std(dim=["chain", "draw"]).values

    bagged_samples = bayesbag(synthetic_data,B=100,mfactor=1)
    theta_bagged_mean = bagged_samples.mean(axis=0)
    theta_bagged_std = bagged_samples.std(axis=0)

    # Plot means
    plt.figure(figsize=(10, 6))
    plt.plot(theta_standard_mean, 'o-', label="Standard Posterior Mean", color='blue')
    plt.plot(theta_bagged_mean, 's--', label="BayesBag Posterior Mean", color='red')
    plt.xlabel("School Index")
    plt.ylabel("Estimated Treatment Effect (θ)")
    plt.title("Standard vs BayesBag Posterior Means (SAT Example)")
    plt.xticks(np.arange(8), labels=schools)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot std deviations
    plt.figure(figsize=(10, 6))
    plt.plot(theta_standard_std, 'o-', label="Standard Posterior Std", color='blue')
    plt.plot(theta_bagged_std, 's--', label="BayesBag Posterior Std", color='red')
    plt.xlabel("School Index")
    plt.ylabel("Posterior Std Dev")
    plt.title("Posterior Spread: Standard vs BayesBag (SAT Example)")
    plt.xticks(np.arange(8), labels=schools)
    plt.legend()
    plt.grid(True)
    plt.show()

    x = np.arange(1, len(y)+1)
    plt.errorbar(x, theta_standard_mean, yerr=0, fmt='o', label='Standard')
    plt.errorbar(x, theta_bagged_mean, yerr=theta_bagged_std, fmt='o', label='BayesBag')
    plt.legend()
    plt.xlabel("School")
    plt.ylabel("Estimated Effect")
    plt.title("Standard vs BayesBag Inference")
    plt.show()

    # Mismatch index
    var_std = np.sum(theta_standard_std ** 2)
    var_bag = np.sum(theta_bagged_std ** 2)
    mismatch_index = 1 - ((2 * var_std) / var_bag) if var_bag > var_std else np.nan
    print(f"MISMATCH INDEX = {mismatch_index:.4f}")

    # Relative Squared Error
    eps = 1e-8
    rse = (theta_bagged_mean - theta_standard_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.mean(rse)
    print(f"RELATIVE SQUARED ERROR = {rse_mean:.4f}")


if __name__ == "__main__":
    main()