import os
import shutil
import json
import math
import unicodedata
from multiprocessing import freeze_support
from datetime import datetime

import numpy as np
from PIL import Image
import pytz
import onnx
from onnxsim import simplify
from onnx_tf.backend import prepare
from onnx import helper
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from HiFuse import HiFuse_Base
from MedViT import MedViT_large

from warnings import filterwarnings
filterwarnings("ignore")


class PathLabelProcessor:
    def __init__(self, base_path, folder_name, pet_type, lesion, symptom):
        self.base_path = base_path
        self.folder_name = folder_name
        self.pet_type = pet_type
        self.lesion = lesion
        self.symptom = symptom

        self.label_images()

    def find_folders_by_name(self):
        matching_folders = []

        for root, dirs, files in os.walk(self.base_path):
            for dir_name in dirs:
                if self.folder_name in dir_name:
                    folder_path = os.path.join(root, dir_name)
                    matching_folders.append(folder_path)

        return matching_folders

    def find_image_json_pairs(self, folder_path):
        image_paths = []
        json_paths = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(("jpg", "png")):
                    image_path = os.path.join(root, file)
                    json_file = f"{os.path.splitext(image_path)[0]}.json"
                    if os.path.isfile(json_file):
                        image_paths.append(image_path)
                        json_paths.append(json_file)

        return image_paths, json_paths

    def is_symptomatic(self, data):
        return data["label"]["label_disease_lv_3"] in self.symptom and data["label"]["label_disease_nm"] == self.lesion

    def label_images(self):
        self.labeled_image_paths = []

        for folder_path in self.find_folders_by_name():
            image_paths, json_paths = self.find_image_json_pairs(folder_path)

            for image_path, json_path in zip(image_paths, json_paths):
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                if data["images"]["meta"]["device"] not in ["스마트폰", "일반카메라"]:
                    continue

                is_symptomatic = self.is_symptomatic(data)
                is_pet_type = self.pet_type in os.path.dirname(image_path).lower()
                label = 1 if is_symptomatic and is_pet_type else 0
                self.labeled_image_paths.append((image_path, label))

        total_cases = len(self.labeled_image_paths)
        asymptomatic_count = sum(label == 0 for _, label in self.labeled_image_paths)
        symptomatic_count = sum(label == 1 for _, label in self.labeled_image_paths)

        print(f"Total cases: {total_cases}")
        print(f"Number of asymptomatic cases: {asymptomatic_count}, Number of symptomatic cases: {symptomatic_count}")

        weight_class_0 = 1.0 / asymptomatic_count
        weight_class_1 = 1.0 / symptomatic_count
        self.class_weights = torch.tensor([weight_class_0, weight_class_1])


class CustomDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        label = torch.tensor(label)

        return image, label


class ImageDataset():
    def __init__(self, data, transform, test_size=None, batch_size=128):
        self.transform = transform
        self.batch_size = batch_size
        self.dataloader = self.create_data_loaders(data, test_size)

    def create_data_loaders(self, data, test_size):
        if test_size:
            train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
            dataset_dict = {"train": train_data, "val": val_data}
        else:
            dataset_dict = {"test": data}

        dataloader = {}
        
        for phase, dataset in dataset_dict.items():
            dataset = CustomDataset(dataset, self.transform[phase])
            dataloader[phase] = DataLoader(dataset, batch_size=self.batch_size,
                                           shuffle=True, num_workers=os.cpu_count())

        return dataloader


class CosineAnnealingWarmUpRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                    1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha.to("cuda") if alpha is not None else None
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", ignore_index=-100)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean().unsqueeze(0)
        elif self.reduction == "sum":
            return focal_loss.sum().unsqueeze(0)
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError("Invalid reduction option")


class ModelTrainer:
    def __init__(self,
                 model,
                 name,
                 dataloader,
                 criterion,
                 optimizer,
                 scheduler):
        self.model = model.to("cuda")
        self.name = name
        self.dataloader = dataloader
        self.criterion = criterion.to("cuda")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_f1_score = 0.0
        self.writer = SummaryWriter(log_dir=f"D:/drharu/ML/diagnosis/runs/{self.name}")

    def calculate_f1_score(self, predicted, labels):
        return f1_score(labels, predicted, average="binary")

    def calculate_auc_roc(self, predicted, labels):
        return roc_auc_score(labels, predicted)

    def compute_metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / total
        return loss, accuracy, predicted
    
    def save_model(self):
        save_path = f"D:/drharu/ML/diagnosis/models/{self.name}"
        
        input_sample = torch.randn(1, 3, img_size, img_size).to("cuda")
        torch.onnx.export(self.model, input_sample, f"{save_path}.onnx", input_names=["input"], opset_version=12)
        
        onnx.save(onnx_model, f"{save_path}.onnx")
         
        onnx_model = onnx.load(f"{save_path}.onnx")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(f"{save_path}.pb")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{save_path}.pb")
        tflite_model = converter.convert()
        
        with open(f"{save_path}.tflite", 'wb') as f:
            f.write(tflite_model)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            try:
                for phase in ["train", "val"]:
                    self.model.train() if phase == "train" else self.model.eval()
                    dataloader = self.dataloader[phase]

                    total_loss = 0.0
                    correct = 0
                    total = 0
                    all_predicted = []
                    all_labels = []

                    for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                        inputs, labels = inputs.to("cuda"), labels.to("cuda")

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            loss, accuracy, predicted = self.compute_metrics(outputs, labels)

                            if phase == "train":
                                if loss.requires_grad:
                                    loss.backward()
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                            total_loss += loss
                            correct += accuracy * labels.size(0)
                            total += labels.size(0)

                            all_predicted.extend(predicted.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    avg_loss = total_loss / len(dataloader)
                    accuracy = correct / total
                    self.writer.add_scalar(f"Loss/{phase}", avg_loss, epoch)
                    self.writer.add_scalar(f"Accuracy/{phase}", accuracy, epoch)

                    if phase == "val":
                        current_f1_score = self.calculate_f1_score(np.array(all_predicted), np.array(all_labels))
                        current_auc_roc = self.calculate_auc_roc(np.array(all_predicted), np.array(all_labels))

                        self.writer.add_scalar("F1 Score/valid", current_f1_score, epoch)
                        self.writer.add_scalar("AUC-ROC/valid", current_auc_roc, epoch)
                        
                        if current_f1_score > self.best_f1_score:
                            self.best_f1_score = current_f1_score
                            self.save_model()

                lr_value = self.scheduler.get_lr()[0]
                self.writer.add_scalar("LearningRate", lr_value, epoch)
                
            except Exception as e:
                model_path = f"D:/drharu/ML/diagnosis/models/{self.name}"
                if os.path.exists(model_path):
                    os.remove(model_path)
                shutil.rmtree(f"D:/drharu/ML/diagnosis/runs/{self.name}", ignore_errors=True)
                raise e

        self.writer.close()


class ModelTester:
    def __init__(self, path, dataloader):
        self.dataloader = dataloader
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.evaluate()

    def classify(self):
        predictions = []
        labels = []

        for inputs, targets in tqdm(self.dataloader):
            inputs = np.array(inputs, dtype=np.float32)
            self.interpreter.set_tensor(self.input_index, inputs)
            self.interpreter.invoke()
            outputs = self.interpreter.get_tensor(self.output_index)

            predicted = np.argmax(outputs, axis=1)
            predictions.extend(predicted)
            labels.extend(targets)

        return predictions, labels

    def evaluate(self):
        predictions, labels = self.classify()
        cm = confusion_matrix(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")

        print("Evaluation Results:")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    freeze_support()
    torch.cuda.empty_cache()
    
    # Choose trian or test
    base_path = "D:/153.반려동물 안구질환 데이터/01.데이터/1.Training"
    # base_path = "D:/153.반려동물 안구질환 데이터/01.데이터/2.Validation"
    
    folder_name = "일반"
    """    
    개: 안검염, 안검종양, 안검내반증, 유루증, 색소침착성각막염, 핵경화, 결막염
    고양이: 안검염, 결막염, 각막부골편, 비궤양성각막염, 각막궤양
    ["유"]

    개: 궤양성각막질환, 비궤양성각막질환
    ["상", "하"]

    개: 백내장
    ["초기", "비성숙", "성숙"]
    """
    pet_type = "개"
    lesion = "안검염"
    symptom = ["유"]
    
    img_size = 256
    
    test_size = 0.1
    batch_size = 128

    # Preprocessing

    processor = PathLabelProcessor(base_path=base_path, folder_name=folder_name, pet_type=pet_type,
                                   lesion=lesion, symptom=symptom)

    data = processor.labeled_image_paths
    class_weights = processor.class_weights
    
    transform = {"train": transforms.Compose([transforms.Resize(img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(degrees=10),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                                                                     hue=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                 "val": transforms.Compose([transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}

    dataloader = ImageDataset(data=data, transform=transform, test_size=test_size, batch_size=batch_size)

    # Modeling

    model = HiFuse_Base(num_classes=2)
    
    model = MedViT_large
    model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=2, bias=True)
    
    model = models.swin_v2_b(weights="DEFAULT")
    model.head = nn.Linear(model.head.in_features, 2)

    for name, param in model.named_parameters():
        if "heads" not in name:
            param.requires_grad = False

    pet_type = "".join([unicodedata.name(char, "Unknown").split()[-1] for char in pet_type])
    lesion = "".join([unicodedata.name(char, "Unknown").split()[-1] for char in lesion])
    start_time = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    name = f"{start_time}_{model.__class__.__name__}_v2_b_{pet_type}_{lesion}"

    criterion = FocalLoss(gamma=2, alpha=class_weights, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=1e-2, T_up=10, gamma=1e-1)

    trainer = ModelTrainer(model=model, name=name, dataloader=dataloader.dataloader,
                           criterion=criterion, optimizer=optimizer, scheduler=scheduler)

    trainer.train(30)
    torch.cuda.empty_cache()

    # Evaluation
    
    processor = PathLabelProcessor(base_path=base_path, folder_name=folder_name, pet_type=pet_type,
                                   lesion=lesion, symptom=symptom)
    
    data = processor.labeled_image_paths
    
    transform = {"test": transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
    
    dataloader = ImageDataset(data=data, transform=transform, test_size=test_size, batch_size=batch_size)
    
    path = r"C:\Users\user\Desktop\pythonProject1\20240206-162718_EfficientNet_v2_m_GAE_ANGEOMYEOM"
    
    ModelTester(path=path, dataloader=dataloader.dataloader["test"])