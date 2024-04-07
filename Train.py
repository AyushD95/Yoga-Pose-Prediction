import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm

# Your imports for model trainingg

# Your imports for ultralytics, YOLO, and other necessary libraries
from ultralytics import YOLO


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Your code for loading the dataset and preprocessing
image_folder = "Test_Data"
model_yolo = YOLO("yolov8x-pose-p6.pt")

data = []

for label in os.listdir(image_folder):
    label_dir = os.path.join(image_folder, label)
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)

        # Extracting keypoint with YOLOv8
        results = model_yolo.predict(img_path, boxes=False, verbose=False)
        for r in results:
            keypoints = r.keypoints.xyn.cpu().numpy()[0]
            keypoints = keypoints.reshape((1, keypoints.shape[0]*keypoints.shape[1]))[0].tolist()
            keypoints.append(img_path)  # insert image path
            keypoints.append(label)     # insert image label

            data.append(keypoints)

total_features = len(data[0])
df = pd.DataFrame(
    data=data,
    columns=[f"x{i}" for i in range(total_features)]
).rename({
    "x34": "image_path", "x35": "label"
}, axis=1)

df = df.dropna()  # delete undetected pose
df = df.iloc[:, 2:]

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)
print("Number of classes:", num_classes)

X = df.drop(["label", "image_path"], axis=1).values
input_length = X.shape[1]
print("Input length:", input_length)

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_tensor = TensorDataset(X_train, y_train)
test_tensor = TensorDataset(X_test, y_test)

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_tensor, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_tensor, batch_size=BATCH_SIZE, shuffle=False)

accuracy_score = Accuracy(task="multiclass", num_classes=num_classes).to(device)
f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)

class YogaClassifier(nn.Module):
    def __init__(self, num_classes, input_length):
        super().__init__()
        self.layer1 = nn.Linear(in_features=input_length, out_features=64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.outlayer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x

input_length = X.shape[1]
model = YogaClassifier(num_classes=num_classes, input_length=input_length).to(device)

optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
loss_fn = nn.CrossEntropyLoss()

epochs = 200

for epoch in tqdm(range(epochs)):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'best.pth')
