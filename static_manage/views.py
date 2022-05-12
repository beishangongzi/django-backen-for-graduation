import os

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

# Create your views here.
from graduation_design import settings
from . import serializers


class FilesViewSet(ViewSet):
    def list(self, request):
        files = os.listdir(settings.MEDIA_ROOT)
        return Response(files)
