functions {
  real normal_ss_log(int N, int P, real y_sq_sum, vector xy_sum,
                     matrix xx_sum, real mu_k, real tau_k, real sigma) {
    real beta_xy;
    real lp;
    vector[P] beta; 
    beta[1] = mu_k;
    beta[2] = tau_k;  // works because P = 2 here
    beta_xy = dot_product(xy_sum, beta);
    lp = -0.5 * (y_sq_sum - 2 * beta_xy + sum(beta * beta' .* xx_sum)) / square(sigma);
    lp -= 0.5 * N * log(square(sigma));
    return lp;
  }
}

data {
  int<lower=0> K;             // number of sites 
  int<lower=0> N;             // total number of observations 
  int<lower=0> P;             // dimensionality of the covariate vector (2 here)
  array[N] real y;            // outcome
  array[N] int ITT;           // treatment assignment
  array[N] int site;          // site membership
}

transformed data {
  array[K] int N_k;                 // sample size per site
  array[K] real y_sq_sum;          // sum of y^2 for each site
  array[K] vector[P] xy_sum;       // sum of y * x for each site
  array[K] matrix[P, P] xx_sum;    // sum of x * x' for each site
  int s;
  vector[P] x;

  N_k = rep_array(0, K);
  y_sq_sum = rep_array(0.0, K);
  xy_sum = rep_array(rep_vector(0.0, P), K);
  xx_sum = rep_array(rep_matrix(0.0, P, P), K);

  x[1] = 1.0;

  for (n in 1:N) {
    s = site[n];
    x[2] = ITT[n];
    N_k[s] += 1;
    y_sq_sum[s] += square(y[n]);
    xy_sum[s] += y[n] * x;
    xx_sum[s] += x * x';
  }
}

parameters {
  real tau;
  real mu;
  vector[K] tau_k;
  vector[K] mu_k;
  real<lower=0> sigma_tau;
  real<lower=0> sigma_mu;
  array[K] real<lower=0> sigma_y_k;
}

model {
  // Priors
  sigma_tau ~ uniform(0, 100000);
  sigma_mu ~ uniform(0, 100000);
  sigma_y_k ~ uniform(0, 100000);
  
  tau ~ normal(0, 1000);
  mu ~ normal(0, 1000);
  tau_k ~ normal(tau, sigma_tau);
  mu_k ~ normal(mu, sigma_mu);

  // Likelihood
  for (k in 1:K) {
    target += normal_ss_log(N_k[k], P, y_sq_sum[k], xy_sum[k], xx_sum[k],
                            mu_k[k], tau_k[k], sigma_y_k[k]);
  }
}

generated quantities {
  real predicted_tau_k = normal_rng(tau, sigma_tau);
  real predicted_mu_k = normal_rng(mu, sigma_mu);
  real signal_noise_ratio_mu = mu / sigma_mu;
  real signal_noise_ratio_tau = tau / sigma_tau;
}
