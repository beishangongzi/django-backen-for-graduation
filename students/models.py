from django.db import models


# Create your models here.
# 注意写模型的时候，每个数据项后面前外不要加comma，否则django不报错，但是不会生成这个数据项
class Student(models.Model):
    SEX_OPTION = (
        (0, "unknown"),
        (1, "male"),
        (2, "female"),
    )

    name = models.CharField(max_length=20, verbose_name="name")
    age = models.SmallIntegerField(verbose_name="age")
    sex = models.SmallIntegerField(choices=SEX_OPTION, verbose_name="sex", default=0)
    classmate = models.CharField(db_column="class", max_length=15, verbose_name="class", default="cs")
    description = models.TextField(null=True, blank=True, verbose_name="bio")

    class Meta:
        db_table = 'db_student'
        verbose_name = "student information"
        verbose_name_plural = verbose_name
