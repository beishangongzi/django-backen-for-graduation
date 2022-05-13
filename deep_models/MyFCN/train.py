import datetime
import os.path

import PIL.Image
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

import deep_models.MyFCN.dataset as dataset
import deep_models.MyFCN.utils as utils
from deep_models.MyFCN.model.FCN import FCN32, FCN16, FCN8
from deep_models.MyFCN.model.FCN2 import my_fcn_resnet50
from deep_models.MyFCN.model.Unet.Unet import Unet
from graduation_design import settings


class Train:
    def __init__(self, dataset_path, model, batch_size, shuffle, mode="train", **kwargs):

        self.dataset = dataset.ObtTrainDataset(dataset_path, mode=mode)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using {self.device}")
        self.model = model.to(self.device)

    def train(self, save_name, save_freq, lr, epoch):
        epoch = int(epoch)
        save_freq = int(save_freq)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=1e-4)
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        for i in range(epoch):
            print("------------{} begin--------------".format(i))
            self.model.train()
            running_loss = 0.0
            j = 0
            for data in dl:
                j += 1
                inputs, target = data
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                target = torch.squeeze(target, 1).long().to(self.device)

                optimizer.zero_grad()

                try:
                    outputs = self.model(inputs)["out"]
                except:
                    outputs = self.model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.cpu().item()
            print(running_loss / j / self.batch_size)
            if (i + 1) % save_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), f"models/{save_name}.pth"))
        torch.save(self.model.state_dict(), os.path.join(os.path.dirname(__file__), f"models/{save_name}_last_.pth"))

    def test(self, save_name):
        """
        this is shit. don't see it.
        :param test_path:
        :param model:
        :return:
        """
        if save_name is None:
            raise ValueError("save name can not be empty")
        save_name = os.path.join(os.path.dirname(__file__), "models", save_name)
        self.model.load_state_dict(torch.load(save_name))
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            cm = torch.zeros((5, 5))
            cm_close = torch.zeros((5, 5))
            cm_erode = torch.zeros((5, 5))
            cm_open = torch.zeros((5, 5))
            cm_dilate = torch.zeros((5, 5))
            for data in dl:
                inputs, names = data
                inputs = inputs.to(self.device)
                try:
                    outputs = self.model(inputs)["out"]
                except:
                    outputs = self.model(inputs)
                outputs = self.model(inputs)
                batch = outputs.size()[0]
                for i in range(batch):
                    output = outputs[i]
                    target = torch.tensor(np.load(
                        os.path.join(os.path.dirname(__file__), "data/obt/testImageMasks/" + names[i])).squeeze())
                    confmat = ConfusionMatrix(num_classes=5)

                    output = output.cpu().numpy()
                    output = np.argmax(output, 0)
                    morphology_x = output
                    cm_in = confmat(torch.tensor(output).reshape([1, -1]), target.reshape([1, -1]))
                    # print(confmat(target.reshape([1, -1]), target.reshape([1, -1])))
                    cm = cm + cm_in
                    output = utils.Utils.to_color(output)

                    pred_name = os.path.join(os.path.dirname(__file__), "data/obt/testImagePreds", names[i] + ".png")
                    PIL.Image.fromarray(output).save(pred_name)
                    ground_truth = np.load(
                        os.path.join(os.path.dirname(__file__), "data/obt/testImageMasks/" + names[i])).squeeze()
                    ground_truth = utils.Utils.to_color(ground_truth)
                    truth_name = os.path.join(os.path.dirname(__file__), "data/obt/testImageMasks", names[i] + ".png")
                    PIL.Image.fromarray(ground_truth).save(truth_name)

                    morphology_close = utils.Morphology.close(morphology_x.astype("uint8"))
                    cm_in_close = confmat(torch.from_numpy(morphology_close).reshape([1, -1]), target.reshape([1, -1]))
                    cm_close += cm_in_close
                    morphology_close = utils.Utils.to_color(morphology_close)
                    morphology_close_name = os.path.join(os.path.dirname(__file__),
                                                         "data/obt/testImageMorphology/close", names[i] + ".png")
                    PIL.Image.fromarray(morphology_close).save(morphology_close_name)

                    morphology_open = utils.Morphology.open(morphology_x.astype("uint8"))
                    cm_in_open = confmat(torch.from_numpy(morphology_open).reshape([1, -1]), target.reshape([1, -1]))
                    cm_open += cm_in_open
                    morphology_open = utils.Utils.to_color(morphology_open)
                    morphology_open_name = os.path.join(os.path.dirname(__file__), "data/obt/testImageMorphology/open",
                                                        names[i] + ".png")
                    PIL.Image.fromarray(morphology_open).save(morphology_open_name)

                    morphology_erode = utils.Morphology.erode(morphology_x.astype("uint8"))
                    cm_in_erode = confmat(torch.from_numpy(morphology_erode).reshape([1, -1]), target.reshape([1, -1]))
                    cm_erode += cm_in_erode
                    morphology_erode = utils.Utils.to_color(morphology_erode)
                    morphology_erode_name = os.path.join(os.path.dirname(__file__),
                                                         "data/obt/testImageMorphology/erode", names[i] + ".png")
                    PIL.Image.fromarray(morphology_erode).save(morphology_erode_name)

                    morphology_dilate = utils.Morphology.dilate(morphology_x.astype("uint8"))
                    cm_in_dilate = confmat(torch.from_numpy(morphology_dilate).reshape([1, -1]),
                                           target.reshape([1, -1]))
                    cm_dilate += cm_in_dilate
                    morphology_dilate = utils.Utils.to_color(morphology_dilate)
                    morphology_dilate_name = os.path.join(os.path.dirname(__file__),
                                                          "data/obt/testImageMorphology/dilate", names[i] + ".png")
                    PIL.Image.fromarray(morphology_dilate).save(morphology_dilate_name)

            with open(f"{settings.MEDIA_ROOT}/res+{os.path.basename(save_name)}.txt", "w", encoding="utf-8") as f:
                f.write("---------pred--------\n")
                f.write(cm.type(torch.int).data.__str__())
                acc = torch.sum(torch.diag(cm.type(torch.int))) / (256 ** 2)
                f.write(f"\nacc={acc.__str__()}\n")

                f.write("\n---------close-------\n")
                f.write(cm_close.type(torch.int).data.__str__())
                acc = torch.sum(torch.diag(cm_close.type(torch.int))) / (256 ** 2)
                f.write(f"\nacc={acc.__str__()}\n")

                f.write("\n---------open-------\n")
                f.write(cm_open.type(torch.int).data.__str__())
                acc = torch.sum(torch.diag(cm_open.type(torch.int))) / (256 ** 2)
                f.write(f"\nacc={acc.__str__()}\n")

                f.write("\n---------erode-------\n")
                f.write(cm_erode.type(torch.int).data.__str__())
                acc = torch.sum(torch.diag(cm_erode.type(torch.int))) / (256 ** 2)
                f.write(f"\nacc={acc.__str__()}\n")

                f.write("\n---------dilate-------\n")
                f.write(cm_dilate.type(torch.int).data.__str__())
                acc = torch.sum(torch.diag(cm_dilate.type(torch.int))) / (256 ** 2)
                f.write(f"\nacc={acc.__str__()}\n")


def run(model_name='', prefix='', mode='', dataset='', backbone='', lr='', epoch='', load_name=None, save_freq='',
        **kwargs):
    models = {"FCN32": FCN32, "FCN16": FCN16, "FCN8": FCN8, "Unet": Unet, "FCN": my_fcn_resnet50}
    print(model_name, "-------")
    if model_name not in models.keys():
        raise ValueError(f"model name must in {models.keys().__str__()}")
    model = models.get(model_name)
    model = model(5, backbone)
    train = Train(dataset, model, 8, True, mode=mode)
    save_name = "-".join([prefix, model_name, backbone, epoch, lr])
    if mode == "train":
        print("train")
        train.train(save_name, save_freq,
                    lr=lr,
                    epoch=epoch)
    else:
        print("access test")
        train.test(load_name)
