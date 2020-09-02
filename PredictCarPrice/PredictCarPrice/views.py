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
    targetval['target'] = prctval

    return render(request, 'index.html', targetval)
    