from django.db import models
class UserInfo(models.Model):
    nid = models.AutoField(primary_key=True)
    ##头像是一个FileField——注意这里必须是“相对路径”，不能是/avatars/这样的绝对路径
    avatar = models.FileField(upload_to='txt/', default='txt/default.jpg')
# Create your models here.
