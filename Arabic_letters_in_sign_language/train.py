import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.asl_cnn import get_model
from tqdm import tqdm
from utils.custom_asl_dataset import SingleFolderASLDataset


def get_data_loaders(data_dir, batch_size=64, val_split=0.1, img_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = SingleFolderASLDataset(data_dir, transform=transform)
    num_classes = len(dataset.class_names)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, num_classes, dataset.class_names

def train(model, device, train_loader, val_loader, criterion, optimizer, epochs, save_path):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with val acc: {best_acc:.4f}")

def evaluate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def main():
    parser = argparse.ArgumentParser(description="Train ASL CNN Model")
    parser.add_argument('--data_dir', type=str, default='Arabic_letters_in_sign_language/data/asl_alphabet_train', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--save_path', type=str, default='Arabic_letters_in_sign_language/outputs/asl_cnn_best.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        args.data_dir, args.batch_size, img_size=args.img_size)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")

    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, device, train_loader, val_loader, criterion, optimizer, args.epochs, args.save_path)
    print(f"Training complete. Best model saved to {args.save_path}")

if __name__ == '__main__':
    main() 