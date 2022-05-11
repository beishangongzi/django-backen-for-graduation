from django.urls import path
from rest_framework.routers import SimpleRouter, DefaultRouter

from . import views

urlpatterns = [
]

router = DefaultRouter()
router.register("backbone", viewset=views.BackboneModelViewSet, basename="backbone")
router.register("segmentation", viewset=views.SegModelViewSet, basename="segmentation")
router.register("savedmodel", viewset=views.SavedModelViewSet, basename="savemodel")
router.register("train-test", viewset=views.TrainAndTestView, basename="train-test")
router.register("test", viewset=views.TestView, basename="test")
router.register("train", viewset=views.TrainView, basename="train")
router.register("Mytest", viewset=views.MyTestView, basename="mytest")
urlpatterns += router.urls
