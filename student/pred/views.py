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
with open("model/rfcmodel.pkl", "rb") as file:
    model = joblib.load(file)

def indexView(request):
    return render(request, "pred/index.html")

def predictionView(request):
    if request.method == "POST":
        studytime = request.POST["studytime"]
        CS201CA = request.POST["201CA"]
        CS201Exam = request.POST["201Exam"]
        CS202CA = request.POST["202CA"]
        CS202Exam = request.POST["202Exam"]
        CS305CA = request.POST["305CA"]
        gender = request.POST["gender"]
        resources = request.POST["resources"]
        extraclass = request.POST.get("extraclass",0)
        MD = request.POST["md"]


        data = np.array([studytime, CS201CA, CS201Exam, CS202CA, CS202Exam, CS305CA, gender, resources, extraclass, MD]).reshape(-1,10)
        # print(data)
        result = model.predict(data)
        # print(result) #debug
        context = {"result" : result}
        return render(request, "pred/result.html", context)
    return render(request, "pred/prediction.html")
