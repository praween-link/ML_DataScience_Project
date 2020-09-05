from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import pickle
with open('model_pickle', 'rb') as f:
    reloadmodel = pickle.load(f)

# Create your views here.

def index(request):
    return render(request,'index.html')

def predicts(request):
    if request.method == 'POST':
        temp = {}
        temp['carmodel'] = request.POST.get('carmodel')
        temp['caryear'] = request.POST.get('caryear')
        temp['cartransmission'] = request.POST.get('cartransmission')
        temp['carmileage'] = request.POST.get('carmileage')
        temp['carfueltype'] = request.POST.get('carfueltype')
        temp['cartax'] = request.POST.get('cartax')
        temp['carmpg'] = request.POST.get('carmpg')
        temp['carengine'] = request.POST.get('carengine')
        
    testData = pd.DataFrame({'x':temp}).transpose()
    prctval = reloadmodel.predict(testData)[0]

    targetval = temp
    
    mol_f = {"0":"A1", "1":"A6", "2":"A4", "3":"A3", "4":"Q3", "5":"Q5", "6":"A5", "7":"S4", "8":"Q2", "9":"A7", "10":"TT", "11":"Q7", "12":"RS6", "13":"RS3", "14":"A8", "15":"Q8", "16":"RS4", "17":"RS5", "18":"R8", "19":"SQ5", "20":"S8", "21":"SQ7", "22":"S3", "23":"S5", "24": "RS7", "25":"A2"}
    trans_f = {"0":"Manual", "1":"Automatic", "2":"Semi-Auto"}
    fuel_f = {"0":"Diesel", "1":"Petrol", "2":"Hybrid"}
    
    if temp['carmodel']:
        targetval['carmodel'] = mol_f[temp['carmodel']]
    if temp['cartransmission']:
        targetval['cartransmission'] = trans_f[temp['cartransmission']]
    if temp['carfueltype']:
        targetval['carfueltype'] = fuel_f[temp['carfueltype']]
    
    targetval['target'] = prctval

    return render(request, 'index.html', targetval)
    
