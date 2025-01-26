import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

def train_one_epoch(model, train_loader, optimizer, criterion, scheduler, step):
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()

        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)

        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

    scheduler.step() 

    return np.mean(loss_list), step

def val_one_epoch(model, val_loader, criterion, epoch):
    model.eval()

    pred = []
    real = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            images, masks = data
            images, masks = images.cuda(non_blocking=True).float(), masks.cuda(non_blocking=True).float()
            out = model(images)

            loss = criterion(out, masks)
            loss_list.append(loss.item())

            real.extend(masks.squeeze(1).cpu().detach().numpy().flatten())
            pred.extend(out.squeeze(1).cpu().detach().numpy().flatten())

    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)

    x_pre = np.where(pred >= 0.5, 1, 0)
    x_real = np.where(real >= 0.5, 1, 0)

    confusion = confusion_matrix(x_real, x_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    Accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    Precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    F1 = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    MIoU = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    Specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    all = MIoU + F1 + Recall + Accuracy + Specificity

    print(f"The Val of Net: Epoch: {epoch}, Loss: {np.mean(loss_list)}, Accuracy: {Accuracy:.4f}, Recall: {Recall:.4f}, Precision: {Precision:.4f}, F1 Score: {F1:.4f}, Mean IoU: {MIoU:.4f}, Specificity: {Specificity:.4f}")

    return np.mean(loss_list), MIoU, all

def test_one_epoch(test_loader,
                    model,
                    criterion):
    model.eval()
    pred = []
    real = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, masks = data
            images, masks = images.cuda(non_blocking=True).float(), masks.cuda(non_blocking=True).float()
            out = model(images)

            loss = criterion(out, masks)
            loss_list.append(loss.item())

            real.extend(masks.squeeze(1).cpu().detach().numpy().flatten())
            pred.extend(out.squeeze(1).cpu().detach().numpy().flatten())

    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)

    x_pre = np.where(pred >= 0.5, 1, 0)
    x_real = np.where(real >= 0.5, 1, 0)

    confusion = confusion_matrix(x_real, x_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    Accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    Recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    Precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
    F1 = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    MIoU = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    Specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0

    print(f"The Test of Net: Loss: {np.mean(loss_list)}, Accuracy: {Accuracy:.4f}, Recall: {Recall:.4f}, Precision: {Precision:.4f}, F1 Score: {F1:.4f}, Mean IoU: {MIoU:.4f}, Specificity: {Specificity:.4f}")

    return np.mean(loss_list)

