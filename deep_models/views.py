import rest_framework.views
from rest_framework.generics import GenericAPIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.views import APIView as View
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
from rest_framework import mixins

from graduation_design.celery import my_sum, run_my_model
from .serializers import SegModelSerializer, \
    BackboneModelSerializer, SaveModelSerializer, \
    TrainAndTestSerializer, TrainAndTestSerializer2
from .models import SegModel, BackboneModel, SavedModel, TrainAndTest
from . import serializers, models


class BackboneModelViewSet(ModelViewSet):
    queryset = BackboneModel.objects.all()
    serializer_class = BackboneModelSerializer


class SegModelViewSet(ModelViewSet):
    queryset = SegModel.objects.all()
    serializer_class = SegModelSerializer


class SavedModelViewSet(ModelViewSet):
    queryset = SavedModel.objects.all()
    serializer_class = SaveModelSerializer


class TrainAndTestView(ModelViewSet):
    queryset = TrainAndTest.objects.all()
    serializer_class = TrainAndTestSerializer2


class TestView(ModelViewSet):
    queryset = TrainAndTest.objects.all()
    serializer_class = serializers.TestSerializer

    def create(self, request, *args, **kwargs):
        validated_data = request.data
        print(validated_data)
        run_my_model.delay(**validated_data)
        return Response({"k": 200})

class TrainView(ModelViewSet):
    queryset = TrainAndTest.objects.all()
    serializer_class = serializers.TrainSeriazlizer

    def create(self, request, *args, **kwargs):
        validated_data = request.data
        print(validated_data)
        run_my_model.delay(**validated_data)
        return Response({"ok": 200})


class MyTestView(ModelViewSet):
    queryset = models.MyTest.objects.all()
    serializer_class = serializers.MyTestSerializer

    def create(self, request, *args, **kwargs):
        print(request.data)
        a = request.data["a"]
        b = request.data["b"]
        my_sum.delay(a, b)
        print(a+b)
        print("hello")
        return Response({"k": 200})
