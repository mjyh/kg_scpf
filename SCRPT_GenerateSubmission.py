# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 14:30:40 2013

@author: Mike
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pylab as pl
from sklearn import ensemble
from sklearn.metrics import mean_squared_error



data_train = pd.read_csv( "train.csv" )
data_test = pd.read_csv( "test.csv" )


#data_test = data_train.ix[196359:]
#data_train = data_train.ix[:196359]

def prepXData( X_raw ):
    X_columns = [
        "Oakland",
        "Chicago",
        "New Haven",
        "Richmond",
        "Has Desc",
        "Age",
        #"Day of Week",
        #"Time of Day",
    ]
    
    X = pd.DataFrame( index = X_raw.index, columns = X_columns )
    
    # Set city indicators
    X[ "Oakland" ] = 0
    X[ "Oakland" ][ (-123 < X_raw.longitude ) & ( X_raw.longitude < -122) & ( 37 < X_raw.latitude ) & ( X_raw.latitude < 38) ] = 1
    
    X[ "Chicago" ] = 0
    X[ "Chicago" ][ (-88 < X_raw.longitude ) & ( X_raw.longitude < -87) & ( 41 < X_raw.latitude ) & ( X_raw.latitude < 43) ] = 1
    
    X[ "New Haven" ] = 0
    X[ "New Haven" ][ (-73 < X_raw.longitude ) & ( X_raw.longitude < -72) & ( 41 < X_raw.latitude ) & ( X_raw.latitude < 42) ] = 1
    
    X[ "Richmond" ] = 0
    X[ "Richmond" ][ (-78 < X_raw.longitude ) & ( X_raw.longitude < -77) & ( 37 < X_raw.latitude ) & ( X_raw.latitude < 38) ] = 1
    
    X[ "Has Desc" ] = 1
    X[ "Has Desc" ][ X_raw[ "description" ].isnull() ] = 0
    
    X[ "Age" ] = np.round( ( pd.datetime( 2013, 5, 1 ) - pd.to_datetime(  X_raw[ "created_time" ] ) ).apply( lambda x: x/np.timedelta64( 86400, 's' ) ) )

    return X
    
X_train = prepXData( data_train )    
X_test = prepXData( data_test )

# Y setup
Y_columns = [ "num_comments", "num_votes", "num_views"  ]
Y_train = data_train[ Y_columns ]
Y_train = np.log( Y_train + 1 )

#Y_CV = data_test[ Y_columns ]
#Y_CV = np.log( Y_CV + 1 )

# fitting

result = pd.DataFrame( index = data_test.id, columns = [ Y_columns ] )

for y_col in Y_columns:
    
    params_dict = {
        "num_comments": { 'n_estimators': 50, 'max_depth': 4, "min_samples_split": 1, "learning_rate": .03, 'loss': 'ls' },
        "num_votes": { 'n_estimators': 40, 'max_depth': 4, "min_samples_split": 1, "learning_rate": .03, 'loss': 'ls' },
        "num_views": { 'n_estimators': 70, 'max_depth': 4, "min_samples_split": 1, "learning_rate": .015, 'loss': 'ls' },
    }
    
    params = params_dict[ y_col ]
    
    model = ensemble.GradientBoostingRegressor( **params )
    
    
    model.fit(X_train, Y_train[ y_col ] )
    
    result[ y_col ] = model.predict( X_test )

#result2 = exp( result ) -1 
#result2.to_csv( 'Sub.csv', index_label = 'id' )