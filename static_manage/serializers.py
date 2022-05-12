from rest_framework import serializers


class StudentSerializer(serializers.Serializer):
    file_name = serializers.CharField()
