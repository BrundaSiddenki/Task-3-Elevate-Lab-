This project implements Simple & Multiple Linear Regression models using the Housing.csv dataset to predict housing prices based on various features. It also explores Polynomial Regression, Feature Scaling, Residual Analysis, and Feature Correlation Visualization for deeper insights.
Tools & Libraries Used
- Python (Programming Language)
- Pandas (Data Handling)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Visualization)
Dataset Description
The dataset includes various features that may impact housing prices:
- price: Target variable (Housing Price)
- area: Size of the property
- bedrooms: Number of bedrooms
- bathrooms: Number of bathrooms
- stories: Number of floors
- mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea: Categorical features
- furnishingstatus: Categorical (unfurnished, semi-furnished, furnished)
Project Steps
- Preprocessing the Data
- Convert categorical features (yes/no, furnished/semi-furnished/unfurnished) into numeric values.
- Apply StandardScaler for feature scaling.
- Train-Test Split
- Divide the dataset into training (80%) and testing (20%) sets using train_test_split.
- Train Models
- Implement Multiple Linear Regression using sklearn.linear_model.LinearRegression.
- Implement Polynomial Regression (degree=2) for handling non-linearity.
- Evaluate Models
- Compute Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score to assess model performance.
- Visualizations & Insights
- Dashboard Plot: Displays actual vs predicted prices for various features.
- Residual Analysis: Checks error distribution.
- Feature Correlation Heatmap: Analyzes how features interact.
Project Structure
Housing_Price_Prediction/
│── Housing.csv        (Dataset)
│── Task3.py   (Main Python script)
│── README.md          (Project Documentation)
How to Run
- Install dependencies
pip install pandas scikit-learn matplotlib seaborn
- Run the script
python housing_price.py
- View the results & plots
Future Improvements
- Try Feature Selection to remove less relevant features.
- Implement Lasso/Ridge Regression for better regularization.
- Test higher-degree polynomial models for further optimization.
