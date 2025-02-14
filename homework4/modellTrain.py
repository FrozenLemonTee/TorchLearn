import os

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from homework4.BinaryClassificationDataset import BinaryClassificationDataset
from homework4.dataSplit import split_data

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwd = os.getcwd()
    print("current working directory: ", cwd)
    src_dir = os.path.join(cwd, 'DogCat_data', 'train')
    print("data src directory: ", src_dir)
    tar_dir = os.path.join(cwd, 'DogCat_data_split')
    if not os.path.exists(tar_dir):
        split_data(src_dir, tar_dir,
                   0.6, 0.2, 0.2)
    dogCat_labels = ['dog', 'cat']
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    dogCat_train = BinaryClassificationDataset(os.path.join(
        cwd, 'DogCat_data_split', 'train'), dogCat_labels, train_transform)
    dogCat_test = BinaryClassificationDataset(os.path.join(
        cwd, 'DogCat_data_split', 'test'), dogCat_labels, valid_transform)
    dogCat_valid = BinaryClassificationDataset(os.path.join(
        cwd, 'DogCat_data_split', 'valid'), dogCat_labels, valid_transform)

    batch_size = 10

    dogCat_train_loader = DataLoader(dogCat_train, batch_size=batch_size, shuffle=True)
    dogCat_test_loader = DataLoader(dogCat_test, batch_size=batch_size, shuffle=True)
    dogCat_valid_loader = DataLoader(dogCat_valid, batch_size=batch_size, shuffle=True)

    my_model = resnet50(pretrained=True)
    num_features = my_model.fc.in_features
    my_model.fc = nn.Linear(num_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_curve = list()
    valid_curve = list()

    MAX_EPOCH = 20
    log_interval = 10
    val_interval = 1

    my_model.to(device)

    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        my_model.train()
        for i, data in enumerate(dogCat_train_loader):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = my_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i + 1, len(dogCat_train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            my_model.eval()
            with torch.no_grad():
                for j, data in enumerate(dogCat_valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = my_model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum()

                    loss_val += loss.item()

                loss_val_epoch = loss_val / len(dogCat_valid_loader)
                valid_curve.append(loss_val_epoch)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(dogCat_valid_loader), loss_val_epoch, correct_val / total_val))
    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(dogCat_train_loader)
    valid_x = torch.arange(1,
                           len(valid_curve) + 1) * train_iters * val_interval - 1  # 由于valid中记录的是epochloss
    # ，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

    my_model.eval()
    total_ = 0
    correct_ = 0
    for i, data in enumerate(dogCat_test_loader):
        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = my_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_ += labels.size(0)
        correct_ += (predicted == labels).squeeze().sum()
        print(f"模型预测为{dogCat_labels[predicted[0]]}，实际为{dogCat_labels[labels]}")
    print(f"Acc:{correct_ / total_}.2%")
