# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def processor_data(train, test):
    
    train_columns = ['Gender',
                     'Age',
                     'Driving_License',
                     'Previously_Insured',
                     'Vehicle_Age',
                     'Vehicle_Damage',
                     'Annual_Premium',
                     'Policy_Sales_Channel',
                     'Vintage'] 
    
    x_train = train[train_columns]
    # Or
    # x_train = train.iloc[:, :11]
    
    x_test = test[train_columns]
        
    y_train = train['Response']
        
    # Preprocessing
    columns_need_dummy = ['Gender',
                          'Driving_License',
                          'Previously_Insured',
                          'Vehicle_Damage',
                          'Vehicle_Age'
                          ]
       
    # Dummmy values     
    x_train = pd.get_dummies(
        x_train,
        columns = columns_need_dummy)
        
    x_test = pd.get_dummies(
        x_test,
        columns = columns_need_dummy)
        
        
        
    # Time to scaling... 
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train}