from django.shortcuts import render
import sys
import warnings
import numpy as np
import joblib
# Create your views here.
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from django.shortcuts import render
#loading model
with open("model/model2.pkl", "rb") as file:
    model = joblib.load(file)

def indexView(request):
    return render(request, "pred/index.html")

def predictionView(request):
    if request.method == "POST":
        gender = request.POST["gender"]
        age = request.POST["age"]
        location = request.POST["location"]
        famsize = request.POST["famsize"]
        traveltime = request.POST["traveltime"]
        studytime = request.POST["studytime"]
        failures = request.POST["failures"]
        paid = request.POST["paid"]
        activities = request.POST["activities"]
        nursery = request.POST["nursery"]
        higher = request.POST["higher"]
        internet = request.POST["internet"]
        freetime = request.POST["freetime"]
        health = request.POST["health"]


        data = np.array([age, traveltime, studytime, failures, freetime, health, gender, location, famsize, 
        paid, activities, nursery, higher, internet]).reshape(-1,14)
        # print(data)
        result = model.predict(data)[0]
        # print(result) #debug
        context = {"result" : result}
        return render(request, "pred/result.html", context)
    return render(request, "pred/prediction.html")
