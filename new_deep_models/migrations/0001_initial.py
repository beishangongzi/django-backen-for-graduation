# Generated by Django 4.0.4 on 2022-05-14 09:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Train',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(help_text='model name', max_length=10)),
                ('train_data', models.CharField(help_text='train data', max_length=100)),
                ('val_data', models.CharField(help_text='validate data', max_length=100)),
                ('batch', models.IntegerField(default=32, help_text='batch size')),
                ('epoch', models.IntegerField(default=100, help_text='epoch default 100')),
                ('lr', models.FloatField(default=0.001, help_text='learning rate')),
                ('decay_rate', models.FloatField(default=0)),
                ('save_prefix', models.CharField(default='a', help_text='prefix of save name', max_length=10)),
                ('flush_secs', models.IntegerField(default=30, help_text='tensorboard flush', max_length=10)),
            ],
        ),
    ]
