from rest_framework.viewsets import ModelViewSet

from .serializers import StudentModelSerializer
from .models import Student


class StudentModelViewSet(ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentModelSerializer
