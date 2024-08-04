
from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import json
import os
# views.py

import subprocess
from django.http import JsonResponse

def start_dalgona_game(request):
    try:
        # Update the path to the script
        subprocess.Popen(['python', 'scripts/dalgonaprocessor2.py'])
        return JsonResponse({'status': 'success', 'message': 'Dalgona game started'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})



def execute_redlight(request):
    try:
        # Update the path to the script
        subprocess.Popen(['python', 'scripts/redlightgreenlight.py'])
        return JsonResponse({'status': 'success', 'message': 'Red Light game started'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

@csrf_exempt
def dalgona_game(request):
    # Render the HTML template for the game
    return render(request, 'game/dalgona.html')

@csrf_exempt
def process_frame(request):
    # This endpoint will handle the real-time processing of frames from OpenCV
    # Read the frame from the request, process it, and return the result
    return HttpResponse(json.dumps({'status': 'success'}), content_type='application/json')

def menu(request):
    return render(request, 'game/menu.html')

def red_light_green_light(request):
    return HttpResponse("Red Light Green Light game")

def dalgona(request):
    return HttpResponse("Dalgona game")

def quit_game(request):
    return render(request, 'game/quit.html')