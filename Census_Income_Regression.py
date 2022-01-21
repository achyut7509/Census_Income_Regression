# IMPORT NECESSARY LIBRARIES

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# IMPORT DATASET

census_income_dataset = pd.read_csv('H:/DATA SCIENCE/Real-Time Datasets/Logistic_Regression/Census_Income_data/adult.data',header = None)

# DATA UNDERSTANDING

    ## ABOUT DATA
       ## 1.  age: continuous.
       ## 2.  workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
       ## 3.  fnlwgt: continuous.
       ## 4.  education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
       ## 5.  education-num: continuous.
       ## 6.  marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
       ## 7.  occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
       ## 8.  relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
       ## 9.  race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
       ## 10. sex: Female, Male.
       ## 11. capital-gain: continuous.
       ## 12. capital-loss: continuous.
       ## 13. hours-per-week: continuous.
       ## 14. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
       ## 15. Income : >50k || <=50K

    ## RENAMING COLUMNS
    
census_income_dataset.columns = ['age','work_class','fnl_wgt','education_degree','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']

    ## INITIAL ANALYSIS

data_shape       = census_income_dataset.shape
data_null_check  = census_income_dataset.isna().sum()
data_dtypes      = census_income_dataset.dtypes
data_description = census_income_dataset.describe(include='all')
data_information = census_income_dataset.info()

    ## DATA TRANSFORMATION
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
census_income_dataset['work_class']        = le.fit_transform(census_income_dataset['work_class'])
census_income_dataset['education_degree']  = le.fit_transform(census_income_dataset['education_degree'])
census_income_dataset['marital_status']    = le.fit_transform(census_income_dataset['marital_status'])
census_income_dataset['occupation']        = le.fit_transform(census_income_dataset['occupation'])
census_income_dataset['relationship']      = le.fit_transform(census_income_dataset['relationship'])
census_income_dataset['race']              = le.fit_transform(census_income_dataset['race'])
census_income_dataset['sex']               = le.fit_transform(census_income_dataset['sex'])
census_income_dataset['native_country']    = le.fit_transform(census_income_dataset['native_country'])
census_income_dataset['income']            = le.fit_transform(census_income_dataset['income'])

    ## CHECKING DATATYPES AFTER TRANSFORMATION

data_dtpes_check       = census_income_dataset.dtypes
data_description_check = census_income_dataset.describe(include='all')

# MODEL BUILDING

X = census_income_dataset.drop(labels = ['income'], axis = 1)
y = census_income_dataset[['income']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 12, stratify = y)

# MODEL TRAINING

from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)

logistic_model.intercept_  #Intercept = -0.00559816

coefficients    = np.array(logistic_model.coef_).T
coefficients_df = pd.DataFrame(coefficients, index = X.columns )

# MODEL TESTING

    ## TRAINING DATA
    
y_predict_train = logistic_model.predict(X_train)
y_predict_train

    ## TEST DATA
    
y_predict_test = logistic_model.predict(X_test)
y_predict_test

# MODEL EVALUATION

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report

    ## TRAINING DATA

print('TRAINING DATA')
print('--------------------------------------------\n')
print('Accuracy Score         :',accuracy_score(y_train,y_predict_train))    # Accuracy = 0.78777
print('Precision Score        :',precision_score(y_train, y_predict_train))  # Precision = 0.62279
print('Recall Score           :',recall_score(y_train,y_predict_train))      # Recall = 0.30080
print('Confusion Matrix       : \n',confusion_matrix(y_train,y_predict_train))
print('Classification Report  :\n',classification_report(y_train, y_predict_train))

    ## TEST DATA

print('TEST DATA')
print('--------------------------------------------\n')
print('Accuracy Score         :',accuracy_score(y_test,y_predict_test))      # Accuracy = 0.78779
print('Precision Score        :',precision_score(y_test, y_predict_test))    # Precision = 0.62102
print('Recall Score           :',recall_score(y_test,y_predict_test))        # Recall = 0.30506
print('Confusion Matrix       : \n',confusion_matrix(y_test,y_predict_test))
print('Classification Report  :\n',classification_report(y_test, y_predict_test))  