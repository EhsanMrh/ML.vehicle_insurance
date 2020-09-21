# Import libraries
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def preprocessing(train, test):
    train_columns = ['Gender',
                 'Age',
                 'Driving_License',
                 'Region_Code',
                 'Previously_Insured',
                 'Vehicle_Age',
                 'Vehicle_Damage',
                 'Annual_Premium',
                 'Policy_Sales_Channel',
                 'Vintage'] 

    x_train = train[train_columns]
    # Or
    # x_train = train.iloc[:, :11]
    
    y_train = train['Response']
    
    # Preprocessing
    columns_need_dummy = ['Gender',
                          'Driving_License',
                          'Region_Code',
                          'Previously_Insured',
                          'Vehicle_Damage',
                          'Vehicle_Age'
                          ]
        
    x_train = pd.get_dummies(
        x_train,
        columns = columns_need_dummy)
    
    x_test = pd.get_dummies(
        test,
        columns = columns_need_dummy)
    
    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train}