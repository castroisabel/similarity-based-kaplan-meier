# Similarity Based Kaplan-Meier Estimator

Author: Isabel de C. Beneyto 

A generalized version of the Kaplan-Meier estimator called the similarity-based Kaplan-Meier. This repository implements an approach that integrates a similarity measure into the traditional Kaplan-Meier formula, enabling weighted covariate assignment for enhanced survival analysis.

## Model 

Consider a data set with observations in the form $`(t_j, \delta_j, \textbf{x}_j)`$, where $`j=1,2,...,n`$. Here $`\textbf{x}_j = (x^1_j,...,x^m_j)^\intercal`$ is a vector of $`m`$ covariates associated with the $`j`$-th observation, $`\delta_j`$ is a binary variable that equals $`1`$ if a failure occurs in the $`j`$-th observation and $`0`$, if it is censored and $`t_j`$ represents the times of failure or censoring for the $`j`$-th observation, depending on whether $`\delta_j=1`$ or $`\delta_j=0`$, respectively, and we define $`t_0 = 0`$. Let $`t_{(i)}`$ be the $`i`$-th smallest positive value in the set $`\{\delta_1t_1, \delta_2t_2, \ldots, \delta_nt_n\}`$, where $`i \in {1, 2, \ldots, n'}`$ and $`n' \leq n`$. Let $`T`$ be the random variable representing the time to failure. Finally, let $`s_w(\textbf{x},\textbf{x}')`$ be a predefined empirical similarity function that measures how similar two covariate vectors $`\textbf{x}`$ and $`\textbf{x}'`$ are (with higher values indicating greater similarity). This function is controlled by the weight vector $`\textbf{w}`$, where each weight corresponds to a specific covariate.

For a given covariate vector $`\textbf{x} = (x^1,...,x^m)^\intercal`$, our aim is to estimate the conditional survival function $`S(t|\textbf{x}) = P(T > t|\textbf{x})`$. 
At this point, we propose using the similarity-based Kaplan-Meier (SBKM) estimator to estimate the conditional survival function. This estimator is defined as follows:

```math
\widehat{S}(t|\textbf{x}) = \prod_{i=1}^{n'}
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

The data was preprocessed (see `notebooks/split_data.ipynb`) to remove columns with NaN values and convert ordinal categorical features into numerical rankings. The dataset was randomly split into $`70\%`$ for training and $`30\%`$ for testing, with $`20\%`$ of the training data used for validation.

The raw data is in the `data/raw` folder, and the processed data, split into training, validation, and test sets, is in the `data/processed` folder.