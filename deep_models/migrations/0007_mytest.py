# Generated by Django 4.0.4 on 2022-05-11 13:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('deep_models', '0006_rename_load_model_trainandtest_load_name_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyTest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('a', models.IntegerField(default=1)),
                ('b', models.IntegerField(default=1)),
            ],
        ),
    ]
