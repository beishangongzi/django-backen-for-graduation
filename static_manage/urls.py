# create by andy at 2022/5/12
# reference:

from django.urls import path, re_path
from rest_framework.routers import SimpleRouter, DefaultRouter
from django.views.static import serve

from graduation_design import settings
from . import views

urlpatterns = [


]

router = DefaultRouter()
router.register("filelist", viewset=views.FilesViewSet, basename="filelist")
urlpatterns += router.urls


if __name__ == '__main__':
    pass
