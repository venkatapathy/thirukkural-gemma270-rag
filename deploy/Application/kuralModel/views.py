
from django.shortcuts import render
from django.http import HttpResponse
from . import model


# Create your views here.
def Test(request):
    return render(request,'home.html')
def Input(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')

        results = model.out(name)

        context = {}
        for i, (details, answer) in enumerate(results, start=1):
            context[f"id{i}"] = details.get("ID", "")
            context[f"t{i}"] = details.get("Kural", "")
            context[f"couplet{i}"] = details.get("Couplet", "")
            context[f"vilakam{i}"] = details.get("Vilakam", "")
            context[f"adhigaram{i}"] = details.get("Adhigaram", "")
            context[f"translit{i}"] = details.get("Transliteration", "")
            context[f"m{i}"] = answer

        return render(request, "result.html", context)

    return render(request, "input.html")