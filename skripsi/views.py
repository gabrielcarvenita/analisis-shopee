from django.http import HttpResponse
from django.shortcuts import render
from . import skripsi_python
from asgiref.sync import sync_to_async
import time

def index(request):    
    if request.method == 'GET':
        context = {
            'judul' :'Analisis',
            
        }
        return render(request, 'index.html', context)
    else:
        data = skripsi_python.hero()
        context = {
            'kalimat': data['kalimat'],
            'data': data['data_review'],
            'grafik': data['save_name'],
            'alasan': data['data_alasan']
        }
        # print(data['data_review'])
        return render(request, 'index.html', context)

