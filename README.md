# Fisher Information Matrix

**A Rust library for computing Fisher information matrices and scores** for common probability distributions used in statistical inference, maximum likelihood estimation, and information geometry.

## Why It Matters

The Fisher information matrix is the backbone of modern statistical inference. It quantifies how much information an observable random variable carries about an unknown parameter. Engineers and data scientists use it to compute the Cramér-Rao lower bound (the theoretical minimum variance of any unbiased estimator), derive Jeffreys priors for Bayesian inference, and understand the geometry of statistical models through information geometry. In machine learning, Fisher information underpins natural gradient descent and the Neural Tangent Kernel.

## How It Works

The library computes Fisher information analytically for each supported distribution by evaluating the expected value of the squared derivative of the log-likelihood. For single-parameter distributions (Bernoulli, Poisson, Exponential), the Fisher information is a scalar — e.g., for `Ber(p)` it is `1/(p(1-p))`, and for `Poi(λ)` it is `1/λ`.

For multi-parameter distributions like the Normal `N(μ, σ²)`, the library returns a flattened 2×2 matrix. The diagonal entries `I_μμ = 1/σ²` and `I_σσ = 2/σ²` measure the curvature of the log-likelihood surface with respect to each parameter, while the off-diagonal entry `I_μσ = 0` reflects the orthogonality of mean and variance in the normal family. The multivariate normal generalizes this to `(d + d²) × (d + d²)` matrices using diagonal covariance assumptions.

The library also implements the **Cramér-Rao bound** (`I⁻¹` gives the minimum achievable variance), the **Jeffreys prior** (`√det(I)` yields a non-informative prior), and an **effective parameter count** that shrinks-regularizes the diagonal via `1 - 1/(1+fᵢ)`. The gamma distribution Fisher matrix uses the trigamma approximation `1/α + 1/(2α²)` for the digamma-related terms.

## Quick Start

```rust
use fisher_information::*;

fn main() {
    // Bernoulli: I(p) = 1 / (p * (1 - p))
    let fi = bernoulli_fisher(0.5);
    println!("Bernoulli FI at p=0.5: {}", fi); // 4.0

    // Normal: returns [I_μμ, I_μσ, I_σμ, I_σσ]
    let normal = normal_fisher(2.0);
    println!("Normal FI: {:?}", normal); // [0.25, 0.0, 0.0, 0.5]

    // Cramér-Rao lower bound
    let cr = cramer_rao_bound(fi);
    println!("CRB: {}", cr); // 0.25

    // Jeffreys prior
    let jp = jeffreys_prior_1d(fi);
    println!("Jeffreys prior: {}", jp); // 2.0
}
```

## API

| Function | Description |
|---|---|
| `normal_fisher(sigma)` | 2×2 Fisher matrix for `N(μ, σ²)` as `[I_μμ, I_μσ, I_σμ, I_σσ]` |
| `bernoulli_fisher(p)` | Scalar Fisher info for `Ber(p)`: `1/(p(1-p))` — **O(1)** |
| `poisson_fisher(lambda)` | Scalar Fisher info for `Poi(λ)`: `1/λ` |
| `exponential_fisher(lambda)` | Scalar Fisher info for `Exp(λ)`: `1/λ²` |
| `gamma_fisher(alpha, beta)` | 2×2 Fisher matrix for `Gamma(α, β)` using trigamma approximation |
| `mvn_fisher_diagonal(d, sigma_diag)` | Diagonal-block Fisher matrix for multivariate normal |
| `cramer_rao_bound(fisher)` | Cramér-Rao lower bound `1/I` — **O(1)** |
| `jeffreys_prior_1d(fisher)` | Jeffreys non-informative prior `√I` |
| `effective_params(diag)` | Effective parameter count via shrinkage |

## Architecture Notes

Part of the SuperInstance mathematical computing ecosystem. This crate provides the statistical primitives consumed by optimization and inference modules. See the [Architecture Guide](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for integration details.

## License

MIT
