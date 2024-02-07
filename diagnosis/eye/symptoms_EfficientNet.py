from datetime import datetime
import json
import math
import os
from multiprocessing import freeze_support
import unicodedata

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm
import pytz

from warnings import filterwarnings

filterwarnings("ignore")


class PathLabelProcessor:
    def __init__(self, base_path, folder_name, devices, symptoms):
        self.base_path = base_path
        self.folder_name = folder_name
        self.devices = devices
        self.symptoms = symptoms
        
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
            for image_file in filter(lambda x: x.lower().endswith(('jpg', 'png')), files):
                image_path = os.path.join(root, image_file)
                json_file = f"{os.path.splitext(image_path)[0]}.json"
                if os.path.isfile(json_file):
                    image_paths.append(image_path)
                    json_paths.append(json_file)

        return image_paths, json_paths

    def label_images(self):
        self.labeled_image_paths = []

        for folder_path in self.find_folders_by_name():
            image_paths, json_paths = self.find_image_json_pairs(folder_path)
            
            for image_path, json_path in zip(image_paths, json_paths):
                with open(json_path, encoding='utf-8') as f:
                    data = json.load(f)

                if data['images']['meta']['device'] not in self.devices:
                    continue
                    
                label = 1 if data['label']['label_disease_lv_3'] in self.symptoms else 0
                self.labeled_image_paths.append((image_path, label))
        
        asymptomatic_count = sum(1 for _, label in self.labeled_image_paths if label == 0)
        symptomatic_count = sum(1 for _, label in self.labeled_image_paths if label == 1)

        print(f'Total cases: {len(self.labeled_image_paths)}')
        print(f'Number of asymptomatic cases: {asymptomatic_count}, Number of symptomatic cases: {symptomatic_count}')
        
        weight_class_0 = 1.0 / asymptomatic_count
        weight_class_1 = 1.0 / symptomatic_count
        self.class_weights = torch.tensor([weight_class_0, weight_class_1])


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)

        return image, label


class ImageDataset():
    def __init__(self,
                 data,
                 transform,
                 test_size,
                 seed,
                 batch_size,
                 shuffle):
        dataset = self.make_dataset(data, transform, test_size, seed)
        self.dataloader = self.make_dataloader(dataset, batch_size, shuffle, os.cpu_count())

    def make_dataset(self, data, transform, test_size=None, seed=42):
        if test_size:
            train_data, val_data = train_test_split(data,
                                                    test_size=test_size,
                                                    random_state=seed)
            dataset_dict = {'train': train_data,
                            'val': val_data}
        else:
            dataset_dict = {'test': data}

        dataset = {k: CustomDataset(v, transform[k])
                   for k, v in dataset_dict.items()}

        return dataset

    def make_dataloader(self, dataset, batch_size, shuffle, num_workers):
        dataloader = {k: DataLoader(dataset=dataset[k],
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=True)
                      for k in dataset.keys()}

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
            param_group['lr'] = lr


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha.to(device) if alpha is not None else None
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option")


class ModelTrainer:
    def __init__(self,
                 model,
                 name,
                 device,
                 dataloader,
                 criterion,
                 optimizer,
                 scheduler):
        self.device = device
        self.model = model.to(self.device)
        self.name = name
        self.dataloader = dataloader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_f1_score = 0.0
        self.writer = SummaryWriter(log_dir=f'D:/drharu/ML/diagnosis/runs/{self.name}')

    def calculate_f1_score(self, predicted, labels):
        return f1_score(labels, predicted, average='binary')

    def calculate_auc_roc(self, predicted, labels):
        return roc_auc_score(labels, predicted)

    def run_epoch(self, epoch, num_epochs):
        for phase in ['train', 'val']:
            self.model.train() if phase == 'train' else self.model.eval()
            dataloader = self.dataloader[phase]

            total_loss = 0.0
            correct = 0
            total = 0
            all_predicted = []
            all_labels = []

            for inputs, labels in tqdm(dataloader, desc=f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}', unit='batch'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    total_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_predicted.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            self.writer.add_scalar(f'Loss/{phase}', avg_loss, epoch)
            self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)

            if phase == 'val':
                current_f1_score = self.calculate_f1_score(np.array(all_predicted), np.array(all_labels))
                current_auc_roc = self.calculate_auc_roc(np.array(all_predicted), np.array(all_labels))

                self.writer.add_scalar('F1 Score/valid', current_f1_score, epoch)
                self.writer.add_scalar('AUC-ROC/valid', current_auc_roc, epoch)

                if current_f1_score > self.best_f1_score:
                    self.best_f1_score = current_f1_score
                    torch.save(self.model, self.name)

        lr_value = self.scheduler.get_lr()[0]
        self.writer.add_scalar('LearningRate', lr_value, epoch)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(epoch, num_epochs)
            self.scheduler.step()

        self.writer.close()


class ModelTester:
    def __init__(self, path, device, dataloader):
        self.device = device
        self.dataloader = dataloader

        self.model = torch.load(path)
        self.model.to(self.device)
        self.evaluate()

    def classify(self):
        self.model.eval()
        predictions = []
        labels = []
        probabilities = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())
                probabilities.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

        return predictions, labels, probabilities

    def calculate_prob_stats(self, probabilities):
        probabilities = np.array(probabilities)
        min_probs = np.min(probabilities)
        max_probs = np.max(probabilities)
        std_probs = np.std(probabilities)
        mean_probs = np.mean(probabilities)

        return min_probs, max_probs, std_probs, mean_probs

    def calculate_percentage(self, value):
        return f'{value * 100:.2f}%'

    def evaluate(self):
        predictions, labels, probabilities = self.classify()
        cm = confusion_matrix(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')

        min_probs, max_probs, std_probs, mean_probs = self.calculate_prob_stats(probabilities)

        print('Evaluation Results:')
        print(f'Confusion Matrix:\n{cm}')
        print(f'Accuracy: {self.calculate_percentage(accuracy)}')
        print(f'F1 Score: {self.calculate_percentage(f1)}')
        print(f'Mean Probability: {self.calculate_percentage(mean_probs)}')
        print(f'Max Probability: {self.calculate_percentage(max_probs)}')
        print(f'Min Probability: {self.calculate_percentage(min_probs)}')
        print(f'Standard Deviation of Probabilities: {std_probs:.4f}')


if __name__ == '__main__':
    freeze_support()

    # Preprocessing

    base_path = 'D:/153.반려동물 안구질환 데이터/01.데이터/1.Training'
    folder_name = '일반'
    devices = ['스마트폰', '일반카메라', '검안경']
    symptoms = ['유', '상', '하', '초기', '비성숙', '성숙']

    processor = PathLabelProcessor(base_path=base_path,
                                folder_name=folder_name,
                                devices=devices,
                                symptoms=symptoms)

    data = processor.labeled_image_paths
    class_weights = processor.class_weights

    transform = {'train': transforms.Compose([transforms.Resize((384, 384)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(degrees=10),
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                'val': transforms.Compose([transforms.Resize((480, 480)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
    test_size = 0.1
    seed = 42
    batch_size = 128
    shuffle = True

    dataloader = ImageDataset(data=data,
                            transform=transform,
                            test_size=test_size,
                            seed=seed,
                            batch_size=batch_size,
                            shuffle=shuffle)

    # Modeling

    model = models.efficientnet_v2_m(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    start_time = datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')
    name = f'D:/drharu/ML/diagnosis/eye/{start_time}_{model.__class__.__name__}_v2_m_symptoms'

    device = torch.device("cuda")
    criterion = FocalLoss(gamma=2, alpha=class_weights, reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=1e-2, T_up=10, gamma=1e-1)

    trainer = ModelTrainer(model=model,
                           name=name,
                           device=device,
                           dataloader=dataloader.dataloader,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler)

    trainer.train(30)
    torch.cuda.empty_cache()

    # Evaluation
    
    base_path = 'D:/153.반려동물 안구질환 데이터/01.데이터/2.Validation'
    folder_name = '일반'
    devices = ['스마트폰', '일반카메라']
    symptom = ['유', '상', '하', '초기', '비성숙', '성숙']

    processor = PathLabelProcessor(base_path=base_path,
                                folder_name=folder_name,
                                devices=devices,
                                symptom=symptom)

    data = processor.labeled_image_paths
    
    transform = {'test': transforms.Compose([transforms.Resize((480, 480)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
    test_size = None
    seed = 42
    batch_size = 128
    shuffle = False
    
    dataloader = ImageDataset(data=data,
                              transform=transform,
                              test_size=test_size,
                              seed=seed,
                              batch_size=batch_size,
                              shuffle=shuffle)
    
    pos_dataloader = ImageDataset(data=[item for item in data if item[1] == 1],
                                  transform=transform,
                                  test_size=test_size,
                                  seed=seed,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
    
    neg_dataloader = ImageDataset(data=[item for item in data if item[1] == 0],
                                  transform=transform,
                                  test_size=test_size,
                                  seed=seed,
                                  batch_size=batch_size,
                                  shuffle=shuffle)
    
    path = 'D:/pythonProject1/20240205-201952_EfficientNet_v2_s_dog_ahngeumyeom.pth'
    device = torch.device("cuda")
    
    ModelTester(path=path, device=device, dataloader=dataloader.dataloader['test'])
    ModelTester(path=path, device=device, dataloader=pos_dataloader.dataloader['test'])
    ModelTester(path=path, device=device, dataloader=neg_dataloader.dataloader['test'])