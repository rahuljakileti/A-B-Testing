# A-B-Testing
A/B Testing &amp; Conversion Modeling with Bootstrapping and Logistic Regression

# A/B Testing Simulation & Streamlit Dashboard

📊 Simulated and analyzed user conversions between two website versions using A/B testing concepts.

## Features
- Simulated 100k user sessions (variant, age, device, time_on_site)
- Bootstrapped confidence intervals for conversion difference (B - A)
- Logistic Regression for conversion prediction
- Streamlit dashboard to visualize results & interact
- Upload your own CSV or generate new test data

## Sample Visualizations
- Conversion by Variant
- Conversion by Device
- Bootstrapped CI Difference (B - A)
- Logistic Regression Report + Feature Importance
- (These auto-generate once the app runs.)

## Key Statistical Concepts
- A/B Testing
- Bootstrapping (resampling CI without normality)
- Confidence Intervals (95%)
- Binary Classification (Logistic Regression)
- EDA and feature importance
  
## Why Simulated Data?
-  To deeply understand statistical inference, we simulated 100,000 user interactions with random conversion behavior for variants A (control) and B (test).
-  This gave full control over assumptions and probabilities — useful for learning and debugging.
-  The code can later be adapted for real data.

## Technologies Used
- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-learn (Logistic Regression)
- Streamlit (Web App)
- Bootstrapping (for statistical inference)

## Future Work
 - Deploy the dashboard on Streamlit Cloud
 - Add session filters (e.g., date range, device selection)
 - Accept real-world datasets from Kaggle
