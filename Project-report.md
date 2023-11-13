# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME HERE
Lakshminarayan Shrivas
## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
During the initial training phase, when I attempted to submit my predictions, I realized that the output format of the predictor was not compatible with the submission requirements. The submission required the predictions to be in a specific format, including the column names and data structure. Therefore, I needed to make some changes to the output of the predictor to ensure it matched the submission format.

### What was the top ranked model that performed?
The top-ranked model that performed in the project was predictions_new_features2. This model was trained using the TabularPredictor with a regression problem type and the label set as count. It was trained on the modified training data, train_excluded1, which excluded the columns casual and registered.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
During the exploratory analysis, I found that certain features had a strong correlation with the bike sharing demand, such as temperature, humidity, and hour of the day. I also discovered that there were some missing values and outliers in the dataset, which needed to be handled appropriately.

### How much better did your model preform after adding additional features and why do you think that is?
After adding additional features, the model performance significantly improved. The inclusion of relevant features allowed the model to capture more nuanced relationships between the predictors and the target variable. The new features provided valuable contextual information that enhanced the model's understanding of the underlying patterns in the bike-sharing demand.

The improvement in performance is reflected in the scores:

Initial Model Score: 1.32878
Model with Additional Features Score: 0.49783

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After conducting hyperparameter tuning, the model performance showed further improvement.

The improvement in performance is reflected in the scores:

Model with Additional Features Score: 0.49783
Model after Hyperparameter Tuning Score: 0.46771
The tuned hyperparameters allowed the model to converge more effectively during training and resulted in better generalization to the test data, leading to a notable reduction in the evaluation metric. This demonstrates the effectiveness of the selected hyperparameters in enhancing the model's performance beyond the initial model with additional features

### If you were given more time with this dataset, where do you think you would spend more time?
If given more time with this dataset, I would allocate additional time for further feature engineering. Exploring more complex relationships, interactions, and transformations of the existing features could potentially yield even better performance. Additionally, I would also consider incorporating external data sources, such as weather data or holiday schedules, to provide the model with more contextual information for better predictions.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
| Model            | hpo1 (learning_rate) | hpo2 (optim_type) | hpo3 (num_trials) | Score      |
| ---------------- | -------------------- | ----------------- | ----------------- | ---------- |
| Initial          | 0.001                | adamw             | -                 | 1.32878    |
| Add Features     | 0.001                | adamw             | -                 | 0.49783    |
| HPO (AutoGluon)  | 0.001                | adamw             | 20                | 0.46771    |


### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
In this project, I used AutoGluon to predict bike sharing demand. Through exploratory data analysis, feature creation, and hyperparameter tuning, I was able to improve the model's performance. Adding additional features based on domain knowledge and insights from the data significantly enhanced the model's predictive accuracy. Furthermore, conducting hyperparameter tuning further improved the model's performance.

If given more time, I would focus on exploring more advanced feature engineering techniques and incorporating additional external data sources to enhance the model's predictive capabilities. Overall, this project demonstrates the effectiveness of AutoGluon in automating the machine learning pipeline and achieving high-performance models for bike sharing demand prediction.
