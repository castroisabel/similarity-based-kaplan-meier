# Similarity Based Kaplan-Meier Estimator

**Author:** Isabel de C. Beneyto 

A generalized version of the Kaplan-Meier (KM) estimator called the similarity-based Kaplan-Meier. This repository implements an approach that integrates a similarity measure into the traditional Kaplan-Meier formula, enabling weighted covariate assignment for enhanced survival analysis.

## Model 

Consider a data set with observations in the form $`(t_j, \delta_j, \textbf{x}_j)`$, where $`j=1,2,...,n`$. Here $`\textbf{x}_j = (x^1_j,...,x^m_j)^\intercal`$ is a vector of $`m`$ covariates associated with the $`j`$-th observation, $`\delta_j`$ is a binary variable that equals $`1`$ if a failure occurs in the $`j`$-th observation and $`0`$, if it is censored and $`t_j`$ represents the times of failure or censoring for the $`j`$-th observation, depending on whether $`\delta_j=1`$ or $`\delta_j=0`$, respectively, and we define $`t_0 = 0`$. Let $`t_{(i)}`$ be the $`i`$-th smallest positive value in the set $`\{\delta_1t_1, \delta_2t_2, \ldots, \delta_nt_n\}`$, where $`i \in {1, 2, \ldots, n'}`$ and $`n' \leq n`$. Let $`T`$ be the random variable representing the time to failure. Finally, let $`s_w(\textbf{x},\textbf{x}')`$ be a predefined empirical similarity function that measures how similar two covariate vectors $`\textbf{x}`$ and $`\textbf{x}'`$ are (with higher values indicating greater similarity). This function is controlled by the weight vector $`\textbf{w}`$, where each weight corresponds to a specific covariate.

For a given covariate vector $`\textbf{x} = (x^1,...,x^m)^\intercal`$, our aim is to estimate the conditional survival function $`S(t|\textbf{x}) = P(T > t|\textbf{x})`$. 
At this point, we propose using the similarity-based Kaplan-Meier (SBKM) estimator to estimate the conditional survival function. This estimator is defined as follows:

```math
\hat{S}(t|\textbf{x}) = \prod_{i=1}^{n'}
\left[1-\frac{\sum_{j=1}^{n} s_w(\textbf{x},\textbf{x}_j) \delta_j \mathbf{1}\{t_j=t_{(i)}\}}
{\sum_{j=1}^{n} s_w(\textbf{x},\textbf{x}_j) \mathbf{1}\{t_j \geq t_{(i)}\}}\right]^{\mathbf{1}\{t_{(i)} \leq t\}}.
```

It is important to note that in the special case where $s_w(\textbf{x},\textbf{x}_j) = 1$ for all $j$, we revert to the classical KM estimator, which estimates the survival curve without considering covariates.


## Code Requirements
To run this project, you need:

- **Python 3.10.12:** Ensure this version of Python is installed on your system.
- **Additional Libraries:** Dependencies are listed in the `requirements.txt` file.

## Setting Up the Environment

Follow these steps to set up your environment and install the required dependencies:

```bash
python 3 -m venv .env
source .env/bin/activate
pip freeze install -r requirements.txt
```

## Dataset: German Credit (CREDIT)

The German Credit dataset, originally provided by Professor Dr. Hans Hofmann from the University of Hamburg, is available in the [UCI Machine Learning Repository](https://doi.org/10.24432/C5NC77). This dataset contains personal and sociodemographic information of various borrowers, including the loan duration in months and the repayment status. 

| **Dataset** | **Sample size** | **Number of covariates** | **Censoring rate** |
|--------------|-----------------|--------------------------|--------------------|
| CREDIT       | 1000            | 17                       | 30.0%             |

Details of the dataset features are outlined in the table below.

| **Variable**              | **Type**       | **Description**                                                                 |
|----------------------------|----------------|---------------------------------------------------------------------------------|
| `duration`                | Numeric        | Duration in months                                                             |
| `full_repaid`             | Categorical    | Specifies whether the loan was fully repaid                                    |
| `age`                     | Numeric        | Borrower's age (in years)                                                      |
| `foreign_worker`          | Categorical    | Indicates whether the borrower is a foreign worker                             |
| `personal_status`         | Categorical    | Gender and marital status                                                      |
| `people_liable`           | Numeric        | Number of dependents                                                           |
| `telephone`               | Categorical    | Indicates whether the borrower has a telephone                                 |
| `employment_years`        | Categorical    | Years (in intervals) at the current job                                        |
| `job`                     | Categorical    | Employment status                                                              |
| `housing`                 | Categorical    | Borrower's housing situation                                                   |
| `present_residence`       | Numeric        | Years at the current residence                                                 |
| `amount`                  | Numeric        | Loan amount                                                                    |
| `installment_rate`        | Numeric        | Percentage of the loan amount charged by the lender to the borrower            |
| `purpose`                 | Categorical    | Reason for obtaining a loan                                                    |
| `checking_account_status` | Categorical    | Status of the checking account                                                 |
| `credit_history`          | Categorical    | Borrower's credit history                                                      |
| `number_of_credits`       | Numeric        | Number of existing credits with this bank                                      |
| `savings_account_status`  | Categorical    | Status of the savings account                                                  |
| `property`                | Categorical    | Type of valuable assets owned by the borrower                                  |

The data was preprocessed (see `notebooks/01.split_data.ipynb`) to remove columns with NaN values and convert ordinal categorical features into numerical rankings. The dataset was randomly split into $`70\%`$ for training and $`30\%`$ for testing, with $`20\%`$ of the training data used for validation.

The raw data is in the `data/raw` folder, and the processed data, split into training, validation, and test sets, is in the `data/processed` folder.

## Evaluation Metrics

Predictive models in survival analysis typically map the set of covariates $`\textbf{x}_i`$ to a risk score associated with a given individual experiencing the event $\eta_i \in \mathbb{R}$, as well as to their survival function $`S(t|\textbf{x}_i)`$. Specifically, we treat these risk scores as measures of the probability of an event occurring for a given individual.

### Concordance Index
The most commonly used evaluation metric for survival models is the concordance index (CI), also known as the *C-index*. The CI is defined as the proportion of all comparable pairs in which predictions and outcomes are concordant. Two samples $i$ and $j$ are comparable if the one with the shorter observed time experienced the event, i.e., $t_j>t_i$​ and $δ_i=1$, where $δ_i$​ is a binary event indicator. Assuming higher $\eta$ values imply shorter survival times, a pair is concordant if $\eta_i>\eta_j$​ and $t_i<t_j$​; otherwise, it is discordant.

Thus, the *C-index* can be expressed as the following probability, conditioned on the relative event order.
```math
CI = P(\eta_i > \eta_j | t_i < t_j)
```

For a perfectly discriminative model, selecting two random comparable subjects $(\eta_i,t_i)$ and $(\eta_j​,t_j​)$, the one with the higher $\eta$ will always have a shorter survival time. Therefore, the concordance index quantifies a model’s discriminative ability, indicating how reliably it ranks individuals by survival time.

This probability can be computed using the following formula:

```math
\hat{CI} = \frac{\sum_{i,j} \mathbf{1}\{t_i <  t_j\} \mathbf{1}\{\eta_i > \eta_j\} \delta_i}{\sum_{i,j} \mathbf{1}\{t_i <  t_j\} \delta_i},
```
where:
- $\eta_i$ is the risk score of individual $i$
- $\mathbf{1}\{t_i <  t_j\} =  1$ if $t_i <  t_j$ otherwise $0$
- $\mathbf{1}\{\eta_i > \eta_j\} =  1$ if $\eta_i > \eta_j$ otherwise $0$

Consequently, $\hat{CI}=1$ indicates the best possible model prediction, while $\hat{CI}=0.5$ corresponds to random guessing.

### Brier Score
Given a dataset with $n$ samples, each sample is represented as $(t_i, \delta_i, \textbf{x}_i)$ where the predicted survival function is $\hat{S}(t|\textbf{x}_i)$.In the absence of censoring, the Brier Score (BS) is computed as:

```math
    BS(t) = \frac{1}{n}  \sum_{i=1}^n  \left( \mathbf{1}\{t_i >  t\} - \hat{S}(t|\textbf{x}_i) \right)^2.
```

For right-censored data, inverse probability of censoring weighting (IPCW) is applied. The Kaplan-Meier estimate of the survival function for censoring times, $\hat{G}(t)$, is given by:

```math
    \hat{G}(t) = \prod_{i=1}^{n} \left(1-\frac{e_i}{g_i}\right)^{(1-\delta_i)\mathbf{1}\{t_i\leq  t\}},
```

where $e_i$​ is the number of censored cases at $t_i$​, and gigi​ is the number of individuals at risk at titi​. Using IPCW, the censored Brier Score is computed as:

```math
    BS(t) = \frac{1}{n} \sum_{i=1}^n \left( \frac{\left(0 - \hat{S}(t|\textbf{x}_i)\right)^2 \mathbf{1}\{t_i \leq  t\} \delta_i}{\hat{G}(t_i)} + \frac{\left(1 - \hat{S}(t|\textbf{x}_i)\right)^2 \mathbf{1}\{t_i >  t\} }{\hat{G}(t)} \right)
```

To assess the overall predictive performance, the Integrated Brier Score (\gls{IBS}) is computed as:

```math
    IBS = \int_{t_1}^{t_{max}} BS(t) d\omega(t), 
```

where $\omega(t) = t/t_{max}$​ is the weighting function. A lower score indicates better predictive accuracy.

The BS is often used to assess calibration since, if a model predicts a $`10\%`$ risk of an event occurring at a given time, the observed frequency in the data should match this percentage for a well-calibrated model. Additionally, the BS is also a measure of discrimination, as it evaluates whether a model can predict risk scores that correctly rank the order of events.

## Results

### Weight Normalization
In this section, we aimed to select the optimal normalization constraint for our weights: $\sum_i w_i = ?$

The following results can be reproduced using the code available in: `notebooks/.ipynb`.

- Similarity Function: **EX**

| $\sum_i w_i$   | Train CI   | Validation CI   | Test CI   | Train IBS   | Validation IBS   | Test IBS   | $w_1$   | $w_2$   | $\sigma (t_m)$   |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| $1$ | 0.7692 | 0.7862 | 0.7786 (0.7438, 0.8101) | 0.1708 | 0.1782 | 0.1763 (0.1508, 0.2058) | 0.2829 | 0.7170 | 0.1603 (0.1381, 0.1827) |
| $10$ | 0.7629 | 0.7831 | 0.7645 (0.7309, 0.7982) | 0.1438 | 0.1501 | 0.1547 (0.1317, 0.1829) | 6.0134 | 3.9865 | 2.7179 (2.1499, 3.3171) |
| $100$ | 0.7700 | 0.7774 | 0.7718 (0.7375, 0.8054) | 0.1109 | 0.1212 | 0.1246 (0.1066, 0.1461) | 75.6461 | 24.3538 | 7.6000 (6.6869, 8.4747) |
| $1000$ | 0.7804 | 0.7630 | 0.7519 (0.7140, 0.7848) | 0.0914 | 0.1282 | 0.1276 (0.1099, 0.1517) | 953.8286 | 46.1713 | 9.4276 (8.3030, 10.3254) |

- Similarity Function: **FR**

| $\sum_i w_i$  | Train CI   | Validation CI   | Test CI   | Train IBS   | Validation IBS   | Test IBS   | $w_1$   | $w_2$   | $\sigma (t_m)$  |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| $1$ | 0.7692 | 0.7818 | 0.7708 (0.7318, 0.8034) | 0.1707 | 0.1781 | 0.1761 (0.1507, 0.2057) | 0.3760 | 0.6239 | 0.1708 (0.1470, 0.1943) |
| $10$ | 0.7666 | 0.7821 | 0.7732 (0.7398, 0.8068) | 0.1585 | 0.1650 | 0.1653 (0.1410, 0.1936) | 5.6410 | 4.3589 | 1.1963 (1.0405, 1.3408) |
| $100$ | 0.7670 | 0.7781 | 0.7734 (0.7405, 0.8055) | 0.1358 | 0.1423 | 0.1447 (0.1239, 0.1701) | 65.0853 | 34.9146 | 3.4317 (3.0623, 3.7347) |
| $1000$ | 0.7782 | 0.7757 | 0.7673 (0.7331, 0.7992) | 0.1083 | 0.1262 | 0.1279 (0.1104, 0.1501) | 763.5755 | 236.4244 | 6.0527 (5.5286, 6.5274) |

### Estimated Failure Time

### Distance Metric

### Invariance of Estimated Weights

### Sampling

### Censoring Rate