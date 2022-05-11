from rest_framework.routers import SimpleRouter, DefaultRouter

from . import views

urlpatterns = [

]

router = DefaultRouter()
router.register("students", viewset=views.StudentModelViewSet, basename="stu")
urlpatterns += router.urls
