import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLCNN(nn.Module):
    def __init__(self, num_classes: int = 29):
        super(ASLCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # For 64x64 input images
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(num_classes: int = 29):
    """
    Returns an instance of the ASLCNN model.
    Args:
        num_classes (int): Number of output classes (default 29 for ASL Alphabet + 'nothing', 'space', 'delete')
    """
    return ASLCNN(num_classes=num_classes) 