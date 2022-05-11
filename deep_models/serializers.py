from rest_framework import serializers


from .models import BackboneModel, SegModel, SavedModel, TrainAndTest, MyTest
from graduation_design.celery import run_my_model

class MyTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyTest
        fields = "__all__"

class SegModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = SegModel
        fields = "__all__"


class BackboneModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = BackboneModel
        fields = "__all__"


class SaveModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = SavedModel
        fields = "__all__"


class TrainAndTestSerializer(serializers.ModelSerializer):
    # mode_operation = (("train", "train"),
    #                   ('val', "val"),
    #                   "test", "test")
    # prefix = serializers.CharField(max_length=10, default='a')
    # model = serializers.CharField(max_length=10, allow_blank=False)
    # load_model = serializers.CharField(max_length=100, allow_blank=True)
    # mode = serializers.ChoiceField(mode_operation)
    # dataset = serializers.CharField(max_length=100)
    # save_freq = serializers.IntegerField(default=20, )
    # epoch = serializers.IntegerField(default=100)
    # lr = serializers.FloatField(default=0.0001)
    class Meta:
        model = TrainAndTest
        fields = "__all__"


class TrainAndTestSerializer2(serializers.Serializer):
    mode_operation = (("train", "train"),
                      ('val', "val"),
                      "test", "test")
    prefix = serializers.CharField(max_length=10, default='a')
    model_name = serializers.CharField(max_length=10, allow_blank=False)
    backbone = serializers.CharField(max_length=10, allow_blank=False)
    load_name = serializers.CharField(max_length=1000, allow_blank=True)
    mode = serializers.ChoiceField(mode_operation)
    dataset = serializers.CharField(max_length=100)
    save_frep = serializers.IntegerField(default=20, )
    epoch = serializers.IntegerField(default=100)
    lr = serializers.FloatField(default=0.0001)



    class Meta:
        model = TrainAndTest


class TestSerializer(serializers.Serializer):
    mode_operation = ("test", "test")
    model_name = serializers.CharField(max_length=10, allow_blank=False)
    backbone = serializers.CharField(max_length=10, allow_blank=False)
    load_name = serializers.CharField(max_length=1000, allow_blank=True)
    mode = serializers.ChoiceField(mode_operation)
    dataset = serializers.CharField(max_length=100)

    def create(self, validated_data):
        print(validated_data)
        run_my_model.delay(**validated_data)
        return validated_data

    class Meta:
        model = TrainAndTest


class TrainSeriazlizer(serializers.Serializer):
    mode_operation = (("train", "train"),
                      ('val', "val"),
                      "test", "test")
    prefix = serializers.CharField(max_length=10, default='a')
    model_name = serializers.CharField(max_length=10, allow_blank=False)
    backbone = serializers.CharField(max_length=10, allow_blank=False)
    mode = serializers.ChoiceField(mode_operation)
    dataset = serializers.CharField(max_length=100)
    save_freq = serializers.IntegerField(default=20, )
    epoch = serializers.IntegerField(default=100)
    lr = serializers.FloatField(default=0.0001)

    def create(self, validated_data):
        print("000000000")
        print(validated_data)
        return {"ee": 40}

    class Meta:
        model = TrainAndTest
