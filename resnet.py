import torch
import torch.nn as nn
from torchvision import models

def create_resnet_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Two output classes: MorningStar and EveningStar
    return model

def train_resnet_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=10):
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds.float() == labels).sum().item()
                total += labels.size(0) * labels.size(1)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / total

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = model.state_dict()

        scheduler.step()

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model
