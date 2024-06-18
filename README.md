Charity Donation Prediction Model


Objective

The objective of this project was to develop a predictive model using machine learning techniques to determine the likelihood of success of charitable donations based on various features from the charity dataset.



Data Preparation

The dataset (charity_data.csv) was imported and processed as follows:
1. Data Cleaning:
    * Columns 'EIN' and 'NAME' were dropped as they were deemed non-beneficial for predictive modeling.
2. Feature Engineering:
    * Categorical variables 'APPLICATION_TYPE' and 'CLASSIFICATION' were analyzed for their value counts.
    * Values with low frequency were grouped into an 'Other' category to reduce dimensionality and noise in the dataset.
3. Data Encoding:
    * Categorical variables were encoded using one-hot encoding (pd.get_dummies) to transform them into a format suitable for machine learning algorithms.
4. Data Splitting:
    * The preprocessed dataset was split into training and testing sets using a 75/25 split ratio.
5. Data Scaling:
    * Numerical features were standardized using StandardScaler to ensure all features were on a similar scale, which is crucial for the performance of deep learning models.
  

      
Model Development

The predictive model was constructed using TensorFlow/Keras, utilizing Keras Tuner for hyperparameter optimization. Here's an overview of the model architecture and tuning process:
1. Model Architecture:
    * The neural network was implemented as a Sequential model with:
        * A dynamic number of hidden layers and neurons per layer chosen by Keras Tuner (Hyperband algorithm).
        * Activation functions (relu, tanh, sigmoid) and learning rates (1e-2, 1e-3, 1e-4) were also selected dynamically.
2. Hyperparameter Tuning:
    * Keras Tuner (Hyperband) was employed to search for the best combination of hyperparameters to maximize validation accuracy (val_accuracy).
    * The tuning process was configured to run for a maximum of 10 epochs per trial, iterating over the hyperband algorithm once.
3. Best Model Selection:
    * After completing the tuning process, the model with the highest validation accuracy was selected.
    * The best hyperparameters were recorded, including activation functions, number of units in each layer, and learning rate.

  
      
Results

Upon completion of the hyperparameter tuning process, the best performing model achieved the following metrics on the test set:
* Accuracy: Approximately 72.77%
* Loss: 0.5570

  
Conclusion

The developed deep learning model demonstrates promising performance in predicting the success of charitable donations based on the provided dataset. Through effective data preprocessing, feature engineering, and hyperparameter tuning, the model was able to achieve a competitive accuracy score. Further enhancements could involve more extensive feature selection, additional tuning of hyperparameters, or exploring ensemble methods to potentially improve performance further.
 
