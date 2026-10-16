# Predicting a Heavy-Tailed Income Density

How much data does it take to learn the tail of an income distribution, and what breaks a model first?

I simulate incomes from a mixture of a lognormal "body" (most people) and a heavy-tailed Lomax "tail" (very high earners). I then train three nonlinear regressors to predict the density from income alone and score them against the true density, which they never see during training.

## Key findings

- **The body is easy, the tail is hard.** With 10,000 observations all three models predict the body to within roughly 2-8% of the true density. The tail depends on how many high earners the model happens to see (about 88 above €250k at n = 10,000, ε = 0.025).
- **More data does not fix a wrong model.** Spline OLS and gradient boosting stop improving beyond n ≈ 10,000. Trees extrapolate as a constant, so boosting stays about 7x off in the far tail even at n = 1,000,000. k-NN keeps improving and reaches about 1.0x.
- **Splines are the most robust with small or noisy data.** With only 1,000 observations the spline is off by about 2.6x in the far tail, versus about 100x for k-NN and about 230x for boosting.
- **Noise hits the local models hardest.** The spline's R² barely moves with noise, while k-NN falls to roughly 0.6 at noise = 1.0, even with a million observations.
- **A heavier tail acts like more data.** Raising the tail share ε from 0.01 to 0.05 at n = 10,000 reduces the far-tail error of the spline from 3.0x to 1.2x. Note that a different ε is a different population, not the same problem made easier.

Far-tail error factor (incomes €1M-€10M; 1.0 = perfect, 2.0 = typically off by a factor of two), ε = 0.025, noise = 0.1, median of 10 runs:

| Model             | n = 1,000 | n = 10,000 | n = 100,000 | n = 1,000,000 |
|-------------------|----------:|-----------:|------------:|--------------:|
| Spline OLS        |      2.64 |       1.40 |        1.39 |          1.38 |
| k-NN              |    103.64 |       3.51 |        1.14 |          1.02 |
| Gradient boosting |    231.92 |       6.31 |        7.06 |          7.12 |

<!-- Add a figure here, e.g. the fitted-curves plot saved from the notebook:
![Fitted curves vs. true density](figures/fitted_curves.png)
-->

## The data

The true density is

```
f(x) = (1 - ε) · LogNormal(x; μ = 10.5, σ = 0.5) + ε · Lomax(x; shape = 1.5, scale = 250,000)
```

Each observation has an `income`, the exact `true_density` f(income), and a noisy `density` = f(income) · exp(s · Z) with Z ~ N(0, 1). The noise level `s` of 0.1 is roughly a 10% error. Models are trained on `income` → `density` only; `true_density` is used afterwards for scoring.

## The models

All models are fitted on log(density).

1. **Spline OLS**: cubic spline on log(income) with 8 quantile knots and linear extrapolation, fitted with ordinary least squares.
2. **k-nearest neighbors**: average of the 15 nearest training incomes.
3. **Gradient boosting**: `HistGradientBoostingRegressor` with 200 iterations and learning rate 0.1.

## Scoring

Besides the usual test-set MSE and R², which mostly measure the body, I report a **factor score** on a log-spaced income grid (3,000 to 10,000,000): how many times too high or too low the prediction is on average. Brackets: body (€10k-€150k) and tail (€250k-€1M and €1M+).

## Experiments

A sweep over tail share ε ∈ {0.01, 0.025, 0.05}, data size n ∈ {1,000, 10,000, 100,000, 1,000,000} and noise ∈ {0, 0.1, 0.3, 1.0}, with 10 fresh datasets per setting and the median score reported.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook predict_density.ipynb
```

The full parameter sweep (section 6 in the notebook) fits 3 models on 480 datasets and takes several minutes. To reduce this, lower `repeats` or the list of sizes in `run_sweep`. Random seeds are fixed, so results are reproducible. The notebook writes an example dataset to `income_data.csv`.

## Repository contents

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `predict_density.ipynb` | Data generator, models, scoring                  |
| `report.pdf`            | Written report (PDF)                             |
| `requirements.txt`      | Python dependencies                              |
| `income_data.csv`       | Example dataset                                  |

## Author
Nooa von Schöneman

www.linkedin.com/in/nooavon

www.nooavon.github.io

## License
MIT