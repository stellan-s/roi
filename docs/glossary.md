# Glossary of Terms

## General Concepts

### **Alpha (α)**
Excess return relative to the market after adjusting for risk. Positive alpha indicates that the strategy adds value.

### **Asset Allocation**
Distribution of capital across asset classes (equities, bonds, commodities, etc.) to balance risk and return.

### **Backtesting**
Evaluating a trading strategy on historical data to estimate its performance before deployment.

### **Beta (β)**
Measures how much an asset moves relative to the market. Beta > 1 implies higher volatility than the market.

### **Bias**
Systematic error in analysis. Example: confirmation bias — the tendency to seek information that confirms existing beliefs.

### **Black Swan**
Extremely rare, high-impact events that are difficult to predict (e.g., COVID-19, the 2008 financial crisis).

## Bayesian Analysis

### **Bayesian Inference**
**Definition**: Statistical method that updates probabilities when new evidence becomes available.
**Formula**: P(H|E) = P(E|H) × P(H) / P(E)
- P(H|E): Posterior probability (after new evidence)
- P(E|H): Likelihood of the evidence given the hypothesis
- P(H): Prior probability (belief before evidence)
- P(E): Evidence (total probability of the evidence)

### **E[r] – Expected Return**
**Definition**: Expected daily return produced by the Bayesian signal combination.
**Unit**: Decimal (0.001 = 0.1% daily return)
**Computation**: Weighted sum of signals adjusted for uncertainty and regime context.

### **Pr(↑) – Probability of Rising**
**Definition**: Probability of a positive price move over the 21-day horizon.
**Unit**: 0–1 (0.65 = 65% probability)
**Usage**: Primary input for buy/sell decisions.

### **Prior Beliefs**
**Definition**: Initial assumptions about how effective each signal is before observing market data.
**Example**: Momentum effectiveness = 0.68 (68% chance that the momentum signal is correct).

### **Posterior**
**Definition**: Updated probability after observing market data.
**Distribution**: Beta distributions model signal effectiveness with parameters α (successes) and β (failures).

## Regime Detection

### **Market Regime**
**Definition**: The prevailing market environment that influences how signals should be interpreted.

#### **Bull Market**
- **Characteristics**: Rising trend, low volatility, optimistic sentiment
- **Signal Adjustment**: Emphasise momentum, maintain trend weight, reduce sentiment weight
- **Allocation**: Up to 100% invested capital

#### **Bear Market**
- **Characteristics**: Falling trend, high volatility, pessimistic sentiment
- **Signal Adjustment**: Down-weight momentum, boost trend and sentiment
- **Allocation**: Maximum 60% invested capital

#### **Neutral Market**
- **Characteristics**: Sideways price action, moderate volatility, mixed signals
- **Signal Adjustment**: Balanced weighting across signals
- **Allocation**: Diversified posture with higher uncertainty tolerance

### **HMM – Hidden Markov Model**
**Definition**: Statistical model for regime transitions when the underlying state is not directly observable.
**Components**:
- Hidden states (Bull/Bear/Neutral)
- Transition probabilities
- Emission probabilities (observations given a state)

### **Regime Persistence**
Probability of remaining in the current regime (prevents over-frequent switching). Default = 0.80.

## Heavy-Tail Risk Modelling

### **Fat Tails**
**Definition**: Higher probability of extreme events than a normal distribution would predict.
**Example**: Equity drawdowns greater than 5% occur more often than under a normal distribution.

### **Student-t Distribution**
**Definition**: Distribution that captures fat tails more realistically than the normal distribution.
**Parameter**: ν (degrees of freedom)
- ν → ∞: Approaches the normal distribution
- ν < 10: Thick tails (typical for equities)
- ν ≈ 3–6: Very heavy tails (high-risk stocks)

### **EVT – Extreme Value Theory**
**Definition**: Statistical framework specifically designed to model tail events.
**Method**: Peaks-Over-Threshold (POT) analyses observations beyond selected thresholds.

### **GPD – Generalized Pareto Distribution**
**Definition**: Distribution used within EVT to model tail exceedances.
**Parameters**:
- ξ (shape): Tail heaviness
- β (scale): Spread of exceedances

### **VaR – Value at Risk**
**Definition**: Maximum expected loss at a given confidence level over a specified time horizon.
**Formula**: P(Loss ≤ VaR) = α (e.g., α = 0.95 for 95% VaR)
**Example**: VaR₉₅% = -5% implies a 5% probability of losing more than 5%.

### **CVaR – Conditional Value at Risk**
**Definition**: Expected loss conditional on losses exceeding VaR.
**Also Called**: Expected Shortfall
**Interpretation**: Average loss in the worst α% of cases.

### **Tail Risk Multiplier**
**Definition**: Ratio between heavy-tail VaR and normal VaR.
**Formula**: Tail Risk Multiplier = VaR_student-t / VaR_normal
**Interpretation**: 2.0× means heavy-tail risk is twice the normal-distribution estimate.

## Monte Carlo Simulation

### **Monte Carlo Method**
**Definition**: Simulation technique that uses random sampling to approximate complex mathematical problems.
**Process**:
1. Draw random samples from the fitted Student-t distribution
2. Apply drift (expected return)
3. Scale by the time horizon
4. Aggregate statistics across simulations

### **Probability Targets**
- **P(return > 0%)**: Probability of a positive return
- **P(return > +20%)**: Probability of gaining more than 20%
- **P(return < -20%)**: Probability of losing more than 20%

### **Percentiles**
- **1st percentile**: Worst 1% outcome
- **99th percentile**: Best 1% outcome

## Portfolio Management

### **Kelly Criterion**
**Definition**: Formula for optimal position sizing given expected return and win probability.
**Formula**: f* = (bp − q) / b
- f*: Optimal capital fraction
- b: Odds (return if the trade wins)
- p: Probability of winning
- q: Probability of losing (1 − p)

### **Risk Parity**
**Definition**: Portfolio construction technique where each position contributes equally to total risk.
**Implementation**: Weightᵢ = (1 / σᵢ) / Σ(1 / σⱼ) where σ represents volatility.

### **Position Sizing in ROI**
```
risk_adjusted_return = E[r] × confidence × regime_stability × tail_risk_penalty
weight = (risk_adjusted_return / Σ risk_adjusted_returns) × total_weight_budget
```

### **Regime Diversification**
Ensures exposure is distributed across market regimes and prevents over-concentration in a single state.

## Statistical Notation Reference

| Symbol | Meaning |
|--------|---------|
| μ      | Expected value (mean) |
| σ      | Standard deviation |
| ν      | Degrees of freedom (Student-t) |
| α      | Confidence level / excess return |
| Σ      | Summation |
| Π      | Product |
| ~      | “Distributed as”, e.g., X ~ N(μ, σ²) |
| 21d    | 21 trading days ≈ 1 month |
| 63d    | 63 trading days ≈ 3 months |
| 252d   | 252 trading days ≈ 1 trading year |
| 12m    | 12 months |

## Decision Labels

- **Buy**: Increase exposure to the asset
- **Sell**: Reduce or exit the position
- **Hold**: Maintain current exposure / no trade

## Probability Notation

- **E[x]**: Expected value of x
- **P(A)**: Probability of event A
- **μ**: Mean
- **σ**: Standard deviation
- **σ²**: Variance
- **df**: Degrees of freedom
- **CI**: Confidence interval
- **Skewness**: Asymmetry of the distribution
- **Kurtosis**: Tail thickness relative to the normal distribution

