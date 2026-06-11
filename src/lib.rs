//! Fisher Information
//!
//! Compute Fisher information matrices and scores for common probability
//! distributions used in statistical inference and information geometry.

/// Fisher information for the normal distribution N(μ, σ²).
/// Returns [I_μμ, I_μσ, I_σμ, I_σσ] as a flattened 2×2 matrix.
pub fn normal_fisher(sigma: f64) -> [f64; 4] {
    let i_mu_mu = 1.0 / (sigma * sigma);
    let i_mu_sigma = 0.0;
    let i_sigma_sigma = 2.0 / (sigma * sigma);
    [i_mu_mu, i_mu_sigma, i_mu_sigma, i_sigma_sigma]
}

/// Fisher information for the Bernoulli distribution Ber(p).
pub fn bernoulli_fisher(p: f64) -> f64 {
    1.0 / (p * (1.0 - p))
}

/// Fisher information for the Poisson distribution Poi(λ).
pub fn poisson_fisher(lambda: f64) -> f64 {
    1.0 / lambda
}

/// Fisher information for the exponential distribution Exp(λ).
pub fn exponential_fisher(lambda: f64) -> f64 {
    1.0 / (lambda * lambda)
}

/// Fisher information matrix for the multivariate normal distribution.
/// Returns a flattened (d+d²) × (d+d²) matrix where d is dimensionality.
pub fn mvn_fisher_diagonal(d: usize, sigma_diag: &[f64]) -> Vec<f64> {
    let n = d + d * d;
    let mut info = vec![0.0; n * n];
    for i in 0..d {
        info[i * n + i] = 1.0 / (sigma_diag[i] * sigma_diag[i]);
    }
    for i in 0..d {
        for j in 0..d {
            let idx = d + i * d + j;
            info[idx * n + idx] = if i == j {
                2.0 / (sigma_diag[i] * sigma_diag[i] * sigma_diag[i] * sigma_diag[i])
            } else {
                1.0 / (sigma_diag[i] * sigma_diag[i] * sigma_diag[j] * sigma_diag[j])
            };
        }
    }
    info
}

/// Compute the Cramér-Rao lower bound for an unbiased estimator variance.
pub fn cramer_rao_bound(fisher: f64) -> f64 {
    1.0 / fisher
}

/// Compute the Jeffreys prior from Fisher information (proportional to sqrt(det(I))).
pub fn jeffreys_prior_1d(fisher: f64) -> f64 {
    fisher.sqrt()
}

/// Fisher information for the gamma distribution Gamma(α, β).
/// Returns [I_αα, I_αβ, I_βα, I_ββ].
pub fn gamma_fisher(alpha: f64, beta: f64) -> [f64; 4] {
    use std::f64::consts::E;
    let digamma_approx = alpha.ln() + 1.0 / (2.0 * alpha); // rough approximation
    let trigamma_approx = 1.0 / alpha + 0.5 / (alpha * alpha);
    let i_aa = trigamma_approx;
    let i_ab = 1.0 / beta;
    let i_bb = alpha / (beta * beta);
    [i_aa, i_ab, i_ab, i_bb]
}

/// Effective number of parameters from the Fisher information.
pub fn effective_params(fisher_matrix_diag: &[f64]) -> f64 {
    fisher_matrix_diag.iter().filter(|&&f| f > 0.0).map(|&f| 1.0 - 1.0 / (1.0 + f)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_fisher() {
        let info = normal_fisher(1.0);
        assert!((info[0] - 1.0).abs() < 1e-10);
        assert!((info[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bernoulli_fisher_symmetry() {
        let f1 = bernoulli_fisher(0.3);
        let f2 = bernoulli_fisher(0.7);
        assert!((f1 - f2).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_fisher() {
        assert!((poisson_fisher(5.0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_cramer_rao() {
        let fi = poisson_fisher(10.0);
        let cr = cramer_rao_bound(fi);
        assert!((cr - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_fisher() {
        assert!((exponential_fisher(2.0) - 0.25).abs() < 1e-10);
    }
}
