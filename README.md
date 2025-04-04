# Sales-Forecasting
This project uses a combination of classical time series techniques and machine learning models to forecast monthly item sales across various shops. After evaluating multiple algorithms, the Random Forest Regressor was selected as the final model due to its strong performance and generalization ability.
## Basic Information
* Person or organization developing model: Robert Bhero, robert.bhero@gwu.edu 
* Model date: March 2025
* Model version: 1.0
* License: MIT
* Model implementation: https://github.com/Robert214-sudo/Sales-Forecasting/blob/main/Untitled.ipynb
## Intended Use

- **Primary intended use:**  
  This model is designed as an educational example for applying **Random Forest regression** to **predict monthly item sales** across different shops and product categories. It demonstrates how machine learning can be used for **time series forecasting** in a retail context, based on features like historical sales, shop performance, and product pricing.

- **Learning objectives:**  
  - Understand the importance of **feature engineering** in time-dependent data.  
  - Compare classical (ARIMA) vs. machine learning approaches for demand prediction.  
  
- **Target audience:**  
  * Primary intended users: Students and those that want to learn about modeling
  * Out of scope use cases: Any use beyond an educational example is out of scope
## Training Data  
*Data Dictionary*

| Column Name           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `date`                | Original transaction date (converted to datetime format)                    |
| `date_block_num`      | Unique month number (used as time index)                                    |
| `shop_id`             | Unique identifier for each shop                                             |
| `item_id`             | Unique identifier for each item/product                                     |
| `item_price`          | Price of the item on the day of sale                                        |
| `item_cnt_day`        | Number of items sold that day (can be fractional for returned items)        |
| `item_cnt_month`      | Aggregated monthly sales count (target variable after grouping)             |
| `item_avg_price`      | Average price of the item across history                                    |
| `shop_avg_sales`      | Average monthly sales for the shop                                          |
| `category_avg_sales`  | Average monthly sales for the item category                                 |
| `item_cnt_month_lag_1`| Item sales count in the previous month                                      |
| `item_cnt_month_lag_2`| Item sales count two months ago                                             |
| `item_cnt_month_lag_3`| Item sales count three months ago                                           |

* Source of training data: Kaggle Predict Future Sales Competition, contact robert.bhero@gwu.edu for more information
  ## Data Splitting

* How training data was divided into training and validation sets:  
  80% training, 20% validation using `train_test_split()` from `sklearn.model_selection`

* Number of rows in training and validation data:
  - **Training rows:** 464,000+ (approx.)
  - **Validation rows:** 116,000+ (approx.)

---

## Test Data
**Source of test data:**  
  Kaggle Predict Future Sales Competition — `test.csv`

* **Number of rows in test data:**  
  21,410 rows (each row represents a unique `shop_id` + `item_id` pair)

## Difference in Columns Between Training and Test Data

The training dataset includes the column `item_cnt_month`, which is the **target variable**, representing the total number of items sold for a given `shop_id` and `item_id` in a specific month.

The **test dataset** does **not** include the `item_cnt_month` column, as it is intended for **generating predictions** to be submitted and evaluated by Kaggle.
## Model Details

### Columns Used as Inputs in the Final Model

The following features were selected and used as inputs to train the final machine learning model (Random Forest Regressor):

- `item_price` – Price of the item during that month  
- `item_avg_price` – Average price of the item over all time  
- `shop_avg_sales` – Average monthly sales per shop  
- `category_avg_sales` – Average monthly sales per item category  
- `item_cnt_month_lag_1` – Sales from one month prior  
- `item_cnt_month_lag_2` – Sales from two months prior  
- `item_cnt_month_lag_3` – Sales from three months prior  

**Note:** Missing lag values were filled with `0` to allow model compatibility.
### 📊 Model Overview

**Column used as target in the final model:** `item_cnt_month`

**Model Type:** Random Forest Regressor  
**Software Used:** Python with scikit-learn  
**scikit-learn Version:** 1.3.0  

**Hyperparameters:**
- `n_estimators`: 100  
- `random_state`: 42  
- `max_depth`: Auto-determined by model  

### 📈 Quantitative Analysis

The model was assessed using the following evaluation metrics:

- **Root Mean Squared Error (RMSE):**  
  - Used as the primary metric for evaluating performance on both validation and test sets.  
  - Lower RMSE indicates better predictive accuracy.

- **Cross-Validation RMSE:**  
  - 5-fold cross-validation was used to assess model generalizability and reduce overfitting.  
  - Average CV RMSE: ~**1.76**

- **Holdout Validation RMSE:**  
  - Final test RMSE after training on 80% of the dataset and validating on the remaining 20%.  
  - Final test RMSE: ~**1.73**

- **Prediction vs. Actual Visualization:**  
  - A plot was used to visually assess how closely the predicted sales matched actual sales values for a sample of the test data.

- **Feature Importance Chart:**  
  - Identified the top features contributing to the model’s predictive performance, including lag features and shop/item average sales.

## Ethical Considerations in Using Predictive Models for Sales Forecasting

While this project is intended for educational and exploratory purposes, it’s important to recognize the ethical implications of applying predictive models in real-world retail or economic decision-making.

Predictive models trained on historical sales data may reflect biases related to consumer behavior, product pricing, regional inequalities, or promotional events that are no longer relevant. For example:

- Models may **overfit past high-performing products or shops**, ignoring newer trends.
- Underperforming or less-promoted items might be **underrepresented**, leading to supply underestimation.
- If used for inventory planning or pricing decisions without caution, such models could **amplify inequalities** in product availability across different regions or consumer groups.

> *As highlighted in AI ethics literature, predictive models can unintentionally reinforce commercial biases and disadvantage certain markets or demographics when trained on narrow or outdated datasets.*

## Limitations of the Project

- Only includes a limited set of features (e.g., doesn’t use promotions, holidays, or external trends)
- Trained purely on historical data — **real-world performance may vary** depending on external factors
- Lag features and price trends may not capture **causal effects** (e.g., marketing, supply chain issues)
- No time-aware validation (e.g., walk-forward or expanding window CV)

## What I Learned

- How to clean and preprocess real-world retail sales data
- How to use **Random Forest** and **XGBoost** for time series regression
- How to apply **cross-validation** and measure forecasting accuracy using RMSE
- How to engineer features like **lag variables** and **group-level averages**
- How to critically think about data-driven decisions and modeling limitations in business contexts

## Future Improvements

- Tune model hyperparameters using **GridSearchCV** or **RandomizedSearchCV**
- Add features like **holidays**, **promotions**, and **economic indicators**
- Incorporate external data (e.g., weather, events, competitor pricing)
- Experiment with time series-aware models (e.g., Prophet, LSTM, or LightGBM with date embeddings)
- Build a **dashboard using Streamlit or Power BI** for forecasting visualization and interaction





