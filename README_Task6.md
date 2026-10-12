# Diamond Price Prediction: Final Data Science Capstone

A machine learning project that predicts the price of a diamond from its attributes, built on the Diamonds dataset (53,790 cleaned records).

## Problem
A retailer needs consistent, fast pricing and a way to flag stones priced out of line. Goal: predict price from carat, cut, colour, clarity and dimensions. Success criteria: R-squared above 0.95 and average error well under 10%.

## Files
- `Diamonds_Price_Prediction_Capstone.ipynb`: full pipeline (cleaning, EDA, feature engineering, modelling, evaluation, error analysis)
- `Diamonds_Capstone_Report.docx`: complete project report
- Presentation: `Diamonds_Capstone_Presentation.pdf` (summary slides)

## Method
1. Cleaned the data (146 duplicates removed before splitting, 20 zero dimensions imputed, 4 impossible records removed).
2. Engineered features: ordinal cut, colour and clarity, plus volume; target transformed to log(price).
3. Trained and compared linear regression, random forest and gradient boosting on an 80/20 split.
4. Evaluated with R-squared, MAE, RMSE and MAPE, plus 5-fold cross-validation, error analysis and permutation importance.

## Results
| Model | R-squared | MAE ($) | MAPE (%) |
|---|---|---|---|
| Linear regression | 0.944 | 451 | 11.4 |
| Random forest | 0.982 | 257 | 6.3 |
| Gradient boosting | 0.982 | 257 | 6.2 |

- 81% of predictions are within 10% of the actual price; 97% within 20%.
- Error grows with size: 5.6% for stones under 0.5 ct, 9.4% above 2 ct.
- Size (carat and dimensions) dominates, then clarity and colour; cut adds very little.

## Recommendations
Use the model as a pricing guide and a flag for out-of-line stones, send large or unusual stones to expert review, and retrain regularly.

## Limitations
Single snapshot with no dates, sales, margin or certification data; limited data for very large stones.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Diamonds_Price_Prediction_Capstone.ipynb
```
The notebook loads the data from a public URL.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook
