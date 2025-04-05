from src.data.make_dataset import load_and_preprocess_data, split_data, scale_data
from src.visualization.visualize import plot_feature_importance
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_logistic_regression, train_random_forest
from src.models.predict_model import evaluate_model, cross_validate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/credit.csv"

    df = load_and_preprocess_data(data_path)
    X, y = create_dummy_vars(df)

    xtrain, xtest, ytrain, ytest = split_data(X, y)

    xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)


    # Train the logistic regression model
    lr_model = train_logistic_regression(xtrain_scaled, ytrain)
    lr_accuracy, lr_cm = evaluate_model(lr_model, xtest_scaled, ytest)
    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"Logistic Regression Confusion Matrix:\n{lr_cm}")

    # Train Random Forest model
    rf_model = train_random_forest(xtrain_scaled, ytrain, n_estimators=2, max_depth=2, max_features=10)
    plot_feature_importance(rf_model, X)
    rf_accuracy, rf_cm = evaluate_model(rf_model, xtest_scaled, ytest)
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Confusion Matrix:\n{rf_cm}")

    # Cross-validation for Logistic Regression
    lr_mean_accuracy, lr_std_accuracy = cross_validate_model(lr_model, xtrain_scaled, ytrain, cv=5)
    print(f"Logistic Regression Cross-Validation Mean Accuracy: {lr_mean_accuracy}")
    print(f"Logistic Regression Cross-Validation Std Accuracy: {lr_std_accuracy}")
    
    # Cross-validation for Random Forest
    rf_mean_accuracy, rf_std_accuracy = cross_validate_model(rf_model, xtrain_scaled, ytrain, cv=5)
    print(f"Random Forest Cross-Validation Mean Accuracy: {rf_mean_accuracy}")
    print(f"Random Forest Cross-Validation Std Accuracy: {rf_std_accuracy}")
