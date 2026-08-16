from sklearn.metrics import (
    r2_score,                          # How much variance the model explains
    explained_variance_score,           # Similar to R² but uses variance directly
    mean_squared_error,                 # Average of squared errors
    mean_absolute_error,                # Average of absolute errors
    median_absolute_error,              # Median of absolute errors (robust to outliers)
    mean_absolute_percentage_error,     # Average percentage error
    max_error                           # Worst single prediction error
)

def regression_report(y_true, y_pred, digits=4):
    """
    Generate a regression report similar to classification_report()

    Parameters:
    -----------
    y_true : array-like
        Actual values (true house prices)
    y_pred : array-like
        Predicted values from your model
    digits : int, default=4
        Number of decimal places to show

    Returns:
    --------
    pandas.DataFrame
        A summary table with regression metrics and their interpretations

    Example:
    --------
    >>> report = regression_report(y_test, y_pred)
    >>> print(report)
    """

    # ============================================================
    # STEP 1: Calculate ALL regression metrics
    # ============================================================

    # 1. R² SCORE (Coefficient of Determination)
    # -----------------------------------------
    # WHAT: Measures how much variance in the data is explained by the model
    # RANGE: 0 to 1 (1 = perfect, 0 = no better than guessing the mean)
    # INTERPRETATION: R² = 0.85 means "model explains 85% of price variation"
    # USE: Best for overall model quality assessment
    # NOTE: Can be negative if model is worse than predicting the mean
    r2 = r2_score(y_true, y_pred)

    # 2. EXPLAINED VARIANCE SCORE
    # ----------------------------
    # WHAT: Similar to R² but uses variance directly (not sum of squares)
    # RANGE: 0 to 1 (1 = perfect)
    # INTERPRETATION: Like R² but less sensitive to mean differences
    # USE: When you care about variance explained specifically
    # NOTE: Usually close to R²; can be slightly different
    explained_var = explained_variance_score(y_true, y_pred)

    # 3. RMSE (Root Mean Squared Error)
    # ---------------------------------
    # WHAT: Square root of average squared errors
    # RANGE: 0 to infinity (lower is better)
    # INTERPRETATION: "Typical error is about RMSE units"
    # USE: Same units as your target (e.g., dollars) - easy to understand
    # NOTE: Penalizes large errors more than small ones
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 4. MSE (Mean Squared Error)
    # ---------------------------
    # WHAT: Average of squared differences between actual and predicted
    # RANGE: 0 to infinity (lower is better)
    # INTERPRETATION: Squared units (e.g., dollars²) - hard to understand
    # USE: As a loss function during model training
    # NOTE: Very sensitive to outliers (large errors get squared)
    mse = mean_squared_error(y_true, y_pred)

    # 5. MAE (Mean Absolute Error)
    # ----------------------------
    # WHAT: Average of absolute differences
    # RANGE: 0 to infinity (lower is better)
    # INTERPRETATION: "Average error is about MAE units"
    # USE: Most intuitive - same units and not heavily penalizing outliers
    # NOTE: Less sensitive to outliers than RMSE
    mae = mean_absolute_error(y_true, y_pred)

    # 6. MEDIAN ABSOLUTE ERROR
    # ------------------------
    # WHAT: Median of absolute differences (middle value when sorted)
    # RANGE: 0 to infinity (lower is better)
    # INTERPRETATION: "Half the predictions are within this error"
    # USE: Very robust to outliers (outliers don't affect median much)
    # NOTE: Good when you have extreme outliers in your data
    median_ae = median_absolute_error(y_true, y_pred)

    # 7. MAPE (Mean Absolute Percentage Error)
    # ----------------------------------------
    # WHAT: Average percentage error
    # RANGE: 0% to 100%+ (lower is better)
    # INTERPRETATION: "Average error is MAPE% of the actual value"
    # USE: Scale-independent, good for comparing across different datasets
    # NOTE: Multiply by 100 to get percentage (e.g., 0.05 = 5%)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # 8. MAX ERROR
    # ------------
    # WHAT: Largest single error the model made
    # RANGE: 0 to infinity (lower is better)
    # INTERPRETATION: "Worst prediction was off by Max Error units"
    # USE: To check for catastrophic failures
    # NOTE: Shows the worst-case scenario for your model
    max_err = max_error(y_true, y_pred)

    # ============================================================
    # STEP 2: Organize all metrics into a dictionary
    # ============================================================

    metrics = {
        'Metric': [
            'R² Score',                      # Overall quality
            'Explained Variance',            # Variance explained
            'RMSE',                          # Error in original units
            'MSE',                           # Squared error (hard to interpret)
            'MAE',                           # Average error
            'Median AE',                     # Median error (robust)
            'MAPE (%)',                      # Percentage error
            'Max Error'                      # Worst case error
        ],
        'Value': [
            r2,                              # 0.8500
            explained_var,                   # 0.8498
            rmse,                            # 25000.00
            mse,                             # 625000000.00
            mae,                             # 18000.00
            median_ae,                       # 15000.00
            mape,                            # 4.2000
            max_err                          # 45000.00
        ]
    }

    # ============================================================
    # STEP 3: Convert to DataFrame and round values
    # ============================================================

    # Create DataFrame with metrics and their values
    report_df = pd.DataFrame(metrics)

    # Round all values to specified decimal places
    report_df['Value'] = report_df['Value'].round(digits)

    # ============================================================
    # STEP 4: Add interpretation for each metric
    # ============================================================

    # Dictionary mapping each metric to its interpretation
    interpretations = {
        'R² Score': 'Higher is better (0-1) - Explained variance proportion',
        'Explained Variance': 'Higher is better (0-1) - Variance captured',
        'RMSE': 'Lower is better - Error in original units (dollars)',
        'MSE': 'Lower is better - Error in squared units (hard to interpret)',
        'MAE': 'Lower is better - Average absolute error in original units',
        'Median AE': 'Lower is better - Robust to outliers (median error)',
        'MAPE (%)': 'Lower is better - Average percentage error',
        'Max Error': 'Lower is better - Worst-case prediction error'
    }

    # Add interpretation column to the DataFrame
    report_df['Interpretation'] = report_df['Metric'].map(interpretations)

    # ============================================================
    # STEP 5: Return the formatted report
    # ============================================================

    return report_df