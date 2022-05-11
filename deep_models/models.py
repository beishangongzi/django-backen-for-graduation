from django.db import models


class BackboneModel(models.Model):
    name = models.CharField(max_length=10, verbose_name="backbone model name")

    def __str__(self):
        return self.name


# Create your models here.
class SegModel(models.Model):
    name = models.CharField(max_length=10, verbose_name="segmentation model name")
    backbone_name = models.ForeignKey(BackboneModel, on_delete=models.CASCADE)

    def __str__(self):
        return "-".join([self.name, self.backbone_name.name])


class SavedModel(models.Model):
    name = models.CharField(max_length=100, verbose_name="the saved mode")


class TrainAndTest(models.Model):
    mode_option = [
        ("val", "val"),
        ("test",  "test"),
        ("train", "train")
    ]
    prefix = models.CharField(max_length=10, default='a', help_text="default a")
    model_name = models.ForeignKey(SegModel, on_delete=models.CASCADE)
    load_name = models.ForeignKey(SavedModel, on_delete=models.CASCADE)
    mode = models.CharField(choices=mode_option, max_length=5)
    dataset = models.CharField(max_length=100)
    save_freq = models.IntegerField()
    epoch = models.IntegerField()
    lr = models.FloatField()

class MyTest(models.Model):
    a = models.IntegerField(default=1)
    b = models.IntegerField(default=1)