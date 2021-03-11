from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request,'index.html',{"predicted":""})

def predict(request):

    pw = str(request.GET['pw'])
    if pw == 'Yes':
        pwv = 1
    else:
        pwv = 2
    
    fc = str(request.GET['fc'])
    if fc == 'Yes':
        fcv = 1
    else:
        fcv = 2
    
    fi = str(request.GET['fi'])
    if fi == 'Yes':
        fiv = 1
    else:
        fiv = 2
    
    jc = str(request.GET['jc'])
    if jc == 'Yes':
        jcv = 1
    else:
        jcv = 2
    
    hi = str(request.GET['hi'])
    if hi == 'Yes':
        hiv = 1
    else:
        hiv = 2

    #Reading the dataframe
    rawdata = staticfiles_storage.path('pep_ds.csv')
    ds = pd.read_csv(rawdata)
    
    # Separating features and target
    x = ds.iloc[:,1:6]
    y = ds.iloc[:,6]
    
    # Split dataframe into train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, random_state = 368)

    # Creating the AI model
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    
    #Prediction
    
    #yet_to_predict = np.array([[exp]])
    #y_pred = regressor.predict(yet_to_predict)

    result = np.array([[pwv, fcv, fiv, jcv, hiv]])
    y_pred = rf.predict(result)

        #rf.predict([[0,1,1,0,1]])
    
    #accuracy = regressor.score(X_test, y_test)
    #accuracy = accuracy*100
    #accuracy = int(accuracy)
    
    return render(request,'index.html',{"predicted":y_pred})
