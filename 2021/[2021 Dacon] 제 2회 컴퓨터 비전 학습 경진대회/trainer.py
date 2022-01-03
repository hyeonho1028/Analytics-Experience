import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm

def train(args, trn_cfg):
    
    train_loader = trn_cfg['train_loader']
    valid_loader = trn_cfg['valid_loader']
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']
    fold_num = trn_cfg['fold_num']
    model_save_path = trn_cfg['model_save_path']

    best_epoch = 0
    best_val_acc = 0.0

    # Train the model
    for epoch in range(args.epochs):
        
        start_time = time.time()
    
        trn_loss, trn_acc = train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = validation(args, model, criterion, valid_loader, device)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        content = "Epoch {} - trn_loss/trn_acc: {:.4f}/{:.4f}  val_loss/val_acc: {:.4f}/{:.4f}, lr: {:.5f}  time: {:.0f}s\n".format(
                    epoch+1, trn_loss, trn_acc, val_loss, val_acc, lr[0], elapsed)
        print(content)

        with open(model_save_path + f'/log_{fold_num}.txt', 'a') as appender:
            appender.write(content + '\n')

        # save model weight
        if val_acc > best_val_acc:
            best_val_acc = val_acc            
            file_save_name = os.path.join(model_save_path, f'fold_epoch_{fold_num}.pth')
            torch.save(model.state_dict(), file_save_name)

        if args.scheduler == 'Plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
    

def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device):

    model.train()
    trn_loss = 0.0
    total_labels = []
    total_outputs = []

    optimizer.zero_grad()

    bar = tqdm(train_loader)
    for images, labels in bar:
        if device:
            images = images.to(device)
            labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

        total_labels.append(labels.cpu().detach().numpy())
        total_outputs.append(outputs.cpu().detach().numpy())

        bar.set_description('loss : % 5f' % (loss.item()))

    total_labels = np.concatenate(total_labels).reshape(-1)
    total_outputs = np.concatenate(total_outputs).reshape(-1)

    epoch_train_loss = trn_loss / len(train_loader)
    total_outputs = np.where(total_outputs>=0.5, 1, 0)
    trn_acc = accuracy_score(total_labels, total_outputs)

    return epoch_train_loss, trn_acc


def validation(args, model, criterion, valid_loader, device):
    
    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    bar = tqdm(valid_loader)
    with torch.no_grad():
        for images, labels in bar:
                        
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            
            total_labels.append(labels.cpu().detach().numpy())
            total_outputs.append(outputs.cpu().detach().numpy())

            bar.set_description('loss : % 5f' % (loss.item()))

    total_labels = np.concatenate(total_labels).reshape(-1)
    total_outputs = np.concatenate(total_outputs).reshape(-1)

    epoch_val_loss = val_loss / len(valid_loader)
    total_outputs = np.where(total_outputs>=0.5, 1, 0)
    val_acc = accuracy_score(total_labels, total_outputs)    
    
    return epoch_val_loss, val_acc

