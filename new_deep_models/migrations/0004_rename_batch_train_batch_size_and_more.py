# Generated by Django 4.0.4 on 2022-05-14 09:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('new_deep_models', '0003_alter_train_batch_alter_train_decay_rate_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='train',
            old_name='batch',
            new_name='batch_size',
        ),
        migrations.RenameField(
            model_name='train',
            old_name='train_data',
            new_name='train_dataset_path',
        ),
        migrations.RenameField(
            model_name='train',
            old_name='val_data',
            new_name='val_dataset_path',
        ),
    ]
