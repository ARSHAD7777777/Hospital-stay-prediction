# Hospital-stay-prediction
### The object of the project was to build a machine learning model which can predict the duration in days a patient stays in a hospital depending upon the values of various features. The given dataset had 18 features including the target variable and 318438 rows or samples. The exploratory data analysis part involved:
#### Checking autocorrelations using pairplot from seaborn API
#### Checking null values
#### Finding unique values of each categorical variable
#### Checking for imbalance in the target variable values
#### From the pairplot it was obvious that none of the features were strongly correlated thus ruling out the option of feature elimination. The features "Age" and "Stay" has got the value 'Nov-20' in some samples. As this value seems irrelevant in comparison with other values of the given features, it was considered a typo and treated as null values.The countplot of the target variable revealed the imbalance in values of the same.

### Data preprocessing involved:

#### Dropping irrelevant variables
#### Splitting the dataset into predictor and target variables
#### Imputing the null values with the most frequent value/mode of each feature using the SimpleImputer from scikitlearn
#### Using SMOTE from imblearn API to generate synthetic data inorder to balance the target variable
#### Mapping the values of some categorical variables into other values to get unique features during one hot encoding
#### One-hot encoding the categorical variables after excluding the discrete quantitative variables and forming a new dataframe with both
#### Encoding the target variable
#### Splitting the preprocessed dataset into training and test data
#### The next step was trying out the various machine learning models for multi-class classification. The followin models were studied for their accuracy
##### RandomForestClassifier
##### XGBClassifier
##### AdaboostClassifier
##### KNeighborsClassifier
### Along with this an artificial neural network was also tried out. It was observed that the XGBoost classifeir model was more efficient than the other models and was consequently used for the project. The RandomizedSearchCV from scikitlearn was used to carry out the Hyper-parameter tuning of XGBClassifier and the best model was selected, tested with the test data and saved using joblib. The complexity of the dataset, particularly after balancing the target variable was a constraint to check the efficiency of more combinations of hyper parameters as it would take a lot of computational power and time.However with the same workflow, much better accuracies can be achieved using a strong processor.
