
functions {
  real normal_ss_log(int N, int P, real y_sq_sum, vector xy_sum,
                     matrix xx_sum, real mu_k, real tau_k, real sigma) {
    real beta_xy;
    real lp;
    vector[P] beta; 
    beta[1] <- mu_k;
    beta[2] <- tau_k;  // this works because P = 2 here
    beta_xy <- dot_product(xy_sum, beta);
    lp <- -.5*(y_sq_sum - 2*beta_xy + sum(beta * beta' .* xx_sum))/sigma^2;
    lp <- lp - .5*N*log(sigma^2);
    return lp;
  }
}


data {
  int<lower=0> K;  // number of sites 
  int<lower=0> N;  // total number of observations 
  int<lower=0> P;  // dimenstionality of the data passed to the likelihood - in this case 2
  real y[N];// outcome variable of interest
  int ITT[N];// intention to treat indicator
  int site[N];//factor variable to split them out into K sites
}


transformed data {
  int N_k[K];  // number of observations from site K
  real y_sq_sum[K];  // sum_i y_{ki}^2
  vector[P] xy_sum[K];  // sum_i y_ki [1, ITT_{ki}]
  matrix[P,P] xx_sum[K];  // sum_i [1, ITT_{ki}] [1, ITT_{ki}]'
  int s;
  vector[P] x;
  // initialize everything to zero
  N_k <- rep_array(0, K);
  y_sq_sum <- rep_array(0.0, K);
  xy_sum <- rep_array(rep_vector(0.0, P), K);
  xx_sum <- rep_array(rep_matrix(0.0, P, P), K);
  // x[1] is always 1
  x[1] <- 1.0;
  for (n in 1:N) {
    s <- site[n];
    x[2] <- ITT[n];
    N_k[s] <- N_k[s] + 1;
    y_sq_sum[s] <- y_sq_sum[s] + y[n]^2;
    xy_sum[s] <- xy_sum[s] + y[n]*x;
    xx_sum[s] <- xx_sum[s] + x*x';
  }
}

parameters {
  real tau;//
  real mu;//
  vector[K] tau_k;// 
  vector[K] mu_k;//
  real<lower=0> sigma_tau;//
  real<lower=0> sigma_mu;//
  real<lower=0> sigma_y_k[K];//


}

transformed parameters {

}

model {
    //  let me try with bounded uniform
  sigma_tau ~ uniform(0,100000);
  sigma_mu ~ uniform(0,100000);
  sigma_y_k ~ uniform(0,100000);

  //  I am hard coding the priors on the hyperparameters here
  tau ~ normal(0,1000); 
  mu ~ normal(0,1000); // one could later insert a more realistic mean but it doesnt matter

  tau_k ~ normal(tau, sigma_tau);

  mu_k ~ normal(mu, sigma_mu);

     for (k in 1:K) {
   
    // increment_normal_ss_lp(N_k[k], P, y_sq_sum[k], xy_sum[k], xx_sum[k], mu_k[k], tau_k[k], sigma_y_k[k]);
    increment_log_prob(normal_ss_log(N_k[k], P, y_sq_sum[k], xy_sum[k], xx_sum[k], mu_k[k], tau_k[k], sigma_y_k[k]));
  }
}
generated quantities{
  real predicted_tau_k; 
  real predicted_mu_k; 
  real signal_noise_ratio_mu;
  real signal_noise_ratio_tau;
  predicted_tau_k <- normal_rng(tau, sigma_tau);
  predicted_mu_k <- normal_rng(mu, sigma_mu);
  signal_noise_ratio_tau <- tau/sigma_tau;
  signal_noise_ratio_mu <- mu/sigma_mu;
}
