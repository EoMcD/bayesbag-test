import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm

# data = gamma.rvs(a=3,scale=2,size=1000)

def data_generation(groups,per_group,seed):
    rng = np.random.default_rng(seed)
    alpha,theta = 3,2
    alpha_sd,theta_sd = 0.3,0.3

    # group level params
    alphas = np.exp(norm.rvs(loc=np.log(alpha),scale=alpha_sd,size=groups,random_state=rng))
    thetas = np.exp(norm.rvs(loc=np.log(theta),scale=theta_sd,size=groups,random_state=rng))

    # synthetic data for each group
    data = np.empty((groups, per_group), dtype=float)
    for g in range(groups):
        data[g, :] = gamma.rvs(a=alphas[g],scale=thetas[g],size=per_group,random_state=rng)
    return data, alphas, thetas

X,a,theta = data_generation(10,100,42)

# verification
sample_mean = X.mean(axis=1)
sample_var = X.var(axis=1,ddof=1)
theo_mean = a*theta
theo_var = a*(theta**2)

print("grp   k      θ      E[X]   mean̂     Var[X]  var̂")
for g in range(len(a)):
    print(f"{g:>3}  {a[g]:6.3f} {theta[g]:6.3f} {theo_mean[g]:6.3f} "
          f"{sample_mean[g]:7.3f} {theo_var[g]:7.3f} {sample_var[g]:7.3f}")

print(f"\nMean abs error of group means: {np.mean(np.abs(sample_mean - theo_mean)):.3f}")
print(f"Mean abs error of group variances: {np.mean(np.abs(sample_var - theo_var)):.3f}")

# z-scores for group means
se_mean = np.sqrt(theo_var / X.shape[1])
z = (sample_mean - theo_mean) / se_mean
print("Mean z  :", z.mean().round(3))
print("Std z   :", z.std(ddof=1).round(3))
print("Max |z| :", np.abs(z).max().round(3))
