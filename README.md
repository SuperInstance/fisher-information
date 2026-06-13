# fisher-information

**Fisher information matrices and Cramér-Rao bounds** for common probability distributions — Normal, Bernoulli, Poisson, Exponential, Gamma, and Multivariate Normal. Includes Jeffreys priors and effective parameter counting for statistical inference and information geometry.

## Why It Matters

Fisher information is the central quantity in mathematical statistics. It measures the amount of information that an observable random variable carries about an unknown parameter. Three major results hinge on it:

1. **Cramér-Rao Bound**: The inverse of Fisher information gives the minimum variance of any unbiased estimator. If you want to estimate a parameter, Fisher information tells you the best you can possibly do.

2. **Jeffreys Prior**: In Bayesian statistics, the square root of Fisher information provides a non-informative prior distribution — invariant under reparameterization.

3. **Information Geometry**: Fisher information defines the Riemannian metric on the statistical manifold of probability distributions. This connects statistics to differential geometry.

This crate provides closed-form Fisher information for the most commonly encountered distributions, along with the Cramér-Rao bound and Jeffreys prior computations.

## How It Works

### Fisher Information Definition

For a parametric family p(x; θ), the Fisher information is:

$$I(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial\theta} \ln p(x;\theta)\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2} \ln p(x;\theta)\right]$$

Under regularity conditions, these two forms are equal. The second form (expected negative second derivative of log-likelihood) is typically easier to compute.

### Normal Distribution N(μ, σ²)

The 2×2 Fisher information matrix for parameters (μ, σ) is:

$$I = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}$$

The off-diagonal element is zero (μ and σ are information-orthogonal for normal distributions). The Cramér-Rao bound for μ estimation is σ²/n — the more data you have or the smaller the variance, the better you can estimate the mean.

### Bernoulli Distribution Ber(p)

$$I(p) = \frac{1}{p(1-p)}$$

This is symmetric around p = 0.5 (maximum information at the most uncertain point) and diverges as p → 0 or p → 1 (certainty about extreme probabilities).

**Cramér-Rao**: The minimum variance of a binomial proportion estimator is p(1-p)/n.

### Poisson Distribution Poi(λ)

$$I(\lambda) = \frac{1}{\lambda}$$

Higher rates → more data → more information. The Cramér-Rao bound is λ/n: the variance of the best unbiased estimator equals the parameter itself divided by sample size.

### Exponential Distribution Exp(λ)

$$I(\lambda) = \frac{1}{\lambda^2}$$

The Cramér-Rao bound is λ²/n.

### Gamma Distribution Gamma(α, β)

$$I_{\alpha\alpha} = \psi'(\alpha), \quad I_{\alpha\beta} = \frac{1}{\beta}, \quad I_{\beta\beta} = \frac{\alpha}{\beta^2}$$

Where ψ'(α) is the trigamma function. The implementation uses a numerical approximation: ψ'(α) ≈ 1/α + 1/(2α²).

### Cramér-Rao Lower Bound

For an unbiased estimator T̂ of parameter θ:

$$\text{Var}(\hat{T}) \geq \frac{1}{I(\theta)}$$

This is the quantum limit of estimation accuracy — no estimator can beat this bound.

### Jeffreys Prior

$$\pi(\theta) \propto \sqrt{\det(I(\theta))}$$

For 1D: `π(θ) ∝ √I(θ)`. Jeffreys priors are reparameterization-invariant — choosing π ∝ √I ensures that the prior doesn't depend on how you parameterize the distribution.

### Effective Number of Parameters

$$n_{\text{eff}} = \sum_i \frac{I_i}{1 + I_i}$$

Each parameter contributes between 0 (I = 0, completely undetermined) and 1 (I → ∞, completely determined) to the effective count. This is useful for model complexity comparison.

### Complexity Analysis

All functions are O(1) except `mvn_fisher_diagonal` which is O(d²) where d = dimensionality (constructing the diagonal of the (d+d²)×(d+d²) matrix).

## Quick Start

```rust
use fisher_information::*;

// Normal distribution Fisher info
let fi = normal_fisher(2.0);
assert!((fi[0] - 0.25).abs() < 1e-10);  // I_μμ = 1/σ² = 1/4

// Cramér-Rao bound for Poisson estimation
let fi_poisson = poisson_fisher(10.0);
let cr_bound = cramer_rao_bound(fi_poisson);
assert!((cr_bound - 10.0).abs() < 1e-10);  // Best possible Var(λ̂) = λ

// Jeffreys prior for Bernoulli
let fi_bern = bernoulli_fisher(0.3);
let prior = jeffreys_prior_1d(fi_bern);
```

## API

### Distribution-Specific Fisher Information
- `normal_fisher(sigma: f64) -> [f64; 4]` — 2×2 matrix for (μ, σ): [I_μμ, I_μσ, I_σμ, I_σσ]
- `bernoulli_fisher(p: f64) -> f64` — I(p) = 1/(p(1−p))
- `poisson_fisher(lambda: f64) -> f64` — I(λ) = 1/λ
- `exponential_fisher(lambda: f64) -> f64` — I(λ) = 1/λ²
- `gamma_fisher(alpha: f64, beta: f64) -> [f64; 4]` — 2×2 matrix for (α, β)
- `mvn_fisher_diagonal(d: usize, sigma_diag: &[f64]) -> Vec<f64>` — Diagonal of MVN Fisher info

### Inference Tools
- `cramer_rao_bound(fisher: f64) -> f64` — Returns 1/I
- `jeffreys_prior_1d(fisher: f64) -> f64` — Returns √I
- `effective_params(fisher_matrix_diag: &[f64]) -> f64` — Σ I_i/(1+I_i)

## Architecture Notes

The Fisher information connects to the γ + η = C conservation framework through information geometry:

- **γ** (gamma) = the statistical manifold's metric tensor (Fisher information matrix)
- **η** (eta) = the dual coordinates (expectation parameters)
- **C** (constant) = the total information in the data (log-likelihood)

The Legendre transform links γ and η: the Fisher-Rao metric provides the duality between natural (θ) and expectation (η) parameterizations. This is the γ + η = C of information geometry.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. Rao, C.R. (1945). "Information and the Accuracy Attainable in the Estimation of Statistical Parameters." *Bull. Calcutta Math. Soc., 37.* — Original definition of Fisher information metric.
2. Cramér, H. (1946). *Mathematical Methods of Statistics.* Princeton. — Cramér-Rao bound.
3. Amari, S. (2016). *Information Geometry and Its Applications.* Springer. — Modern information geometry.
4. Jeffreys, H. (1946). "An Invariant Form for the Prior Probability in Estimation Problems." *Proc. Royal Society A, 186.*
5. Ly, A., et al. (2017). "A Tutorial on Fisher Information." *J. Mathematical Psychology, 80.* — Comprehensive modern tutorial.

## License

MIT
