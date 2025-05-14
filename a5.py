import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision.transforms import v2
from captum.attr import Occlusion
from captum.attr import visualization as viz
import torchvision.transforms as transforms
import torchvision.models as models

### ----------------------- Understanding the dataset ----------------------- ###

def print_sample_images():
    folder = ImageFolder('val', transform=torchvision.transforms.ToTensor())
    loader = DataLoader(folder, batch_size=8, shuffle=True)

    Xexamples, Yexamples = next(iter(loader))
    print(Xexamples.shape) # check image size

    #saving figure with 8 example images
    for i in range(8):
        plt.subplot(2,4,i+1)  
        img = Xexamples[i].numpy().transpose(1, 2, 0)    
        plt.imshow(img, interpolation='none')
        plt.title('NV' if Yexamples[i] else 'MEL')
        plt.xticks([])
        plt.yticks([])

    plt.savefig("sample_images.png")
    plt.close()

print_sample_images()

### ------------------------- Creating Dataloaders -------------------------- ###

def create_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = ImageFolder('train', transform=transform)
    val_dataset = ImageFolder('val', transform=transform)
    test_dataset = ImageFolder('test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # no shuffling on validation set
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    return train_loader, val_loader, test_loader

from torchvision.transforms import v2
import torch

# Data Augmentation
def create_aug_dataloaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    aug_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    aug_train_dataset = ImageFolder('train', transform=aug_transform)
    aug_val_dataset = ImageFolder('val', transform=transform)

    aug_train_loader = DataLoader(aug_train_dataset, batch_size=8, shuffle=True)
    aug_val_loader = DataLoader(aug_val_dataset, batch_size=8, shuffle=False)
    return aug_train_loader, aug_val_loader

### ---------------------------- Baseline Model ----------------------------- ###

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128*16*16, 100) # after 3 poolings 128x128 becomes 16x16
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(100, 2) # 2 possible output classes

    def forward(self, X):
        c1 = self.conv1(X)
        fm1 = F.max_pool2d(F.relu(c1), 2)
        c2 = self.conv2(fm1)
        fm2 = F.max_pool2d(F.relu(c2), 2)
        c3 = self.conv3(fm2)
        fm3 = F.max_pool2d(F.relu(c3), 2)
        fl = torch.flatten(fm3, start_dim=1)
        h1 = F.relu(self.fc(fl))
        h1 = self.dropout(h1)
        out = self.output(h1)
        return out
       
### ----------------------------- Training Loop ----------------------------- ###

def train_classifier(model, train_data, val_data, hyperparams):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])    
   
    # cross-entropy loss
    loss_func = torch.nn.CrossEntropyLoss()
     
    val_acc_history = []
    train_acc_history = []
    best_val_acc = 0
    best_model_state = None

    for epoch in range(hyperparams['n_epochs']):

        # set the model in training mode
        model.train()
       
        loss_sum = 0

        for Xbatch, Ybatch in tqdm(train_data):

            # the output should be of the shape (batch_size, 2), since we have 2 classes
            outputs = model(Xbatch)

            loss = loss_func(outputs, Ybatch)

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            loss_sum += loss.item()

        # set model in evaluation mode
        model.eval()
        with torch.no_grad(): # for faster computations and less memory usage
            # compute accuracy on validation data
            val_acc = predict_and_evaluate(model, val_data)
            train_acc = predict_and_evaluate(model, train_data)
               
        mean_loss = loss_sum / len(train_data)

        val_acc_history.append(val_acc)
        train_acc_history.append(train_acc)
       
        print(f'Epoch {epoch+1}: train loss = {mean_loss:.4f}, val acc = {val_acc:.4f}')
   
        # save the model if it is the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    model_name = f"{model.__class__.__name__}_best.pth"
    torch.save(best_model_state, model_name)
    print(f"Best model saved with val acc {best_val_acc:.4f} as '{model_name}'")

    return val_acc_history, train_acc_history
       

### ------------------------------- Evaluation ------------------------------ ###

def predict_and_evaluate(model, data):
    all_true_labels = []
    all_pred_labels = []
   
    for Xbatch, Ybatch in data:
        outputs = model(Xbatch)
        predictions = outputs.argmax(dim=1)
       
        all_true_labels.extend(Ybatch.numpy())
        all_pred_labels.extend(predictions.numpy())

    plot_confusion_matrix(all_true_labels, all_pred_labels)

    return accuracy_score(all_true_labels, all_pred_labels)


def plot_confusion_matrix(Y, pred):
    cm = confusion_matrix(Y, pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, cmap= "Blues", annot=True, fmt="d", linewidths=3, xticklabels=["NV", "MEL"], yticklabels=["NV", "MEL"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion Matrix", fontsize=16)
   
    # saving the confusion matrix as an image
    plt.savefig("conf_matrix.png")
    plt.close()


def plot_misclassified_images_mlp(model, img_dataloader, feature_dataloader, n_images=6):
    model.eval()
    error_images = []
    true_labels = []
    pred_labels = []

    img_iter = iter(img_dataloader)  # load original images
    feature_iter = iter(feature_dataloader)

    with torch.no_grad():
        for (img_batch, img_labels), (feature_batch, feature_labels) in zip(img_iter, feature_iter):
            predictions = model(feature_batch).argmax(dim=1)

            for i in range(8):
                if len(error_images) >= n_images:
                    break
                if predictions[i] != feature_labels[i]:
                    error_images.append(img_batch[i]) 
                    true_labels.append(feature_labels[i].item())
                    pred_labels.append(predictions[i].item())

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Misclassified images (MLP)")

    classes = ["MEL", "NV"]
    for i in range(n_images):
        plt.subplot(2, 3, i + 1)
        img = error_images[i].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        plt.imshow(img)
        plt.title(f"True: {classes[true_labels[i]]}, Pred: {classes[pred_labels[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_images_mlp.png") 
    plt.close()


### -------------------------- Tuning Learnig Rate -------------------------- ###

def tune_learningrate(model_class, train_data, val_data, model_name="model_name"):
    lr_list = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    accuracy_dict = {}
    best_accuracy = 0.0
    best_lr = None

    for lr in lr_list:
        torch.random.manual_seed(2)
        model_instance = model_class()
        val_acc_history, train_acc_history = train_classifier(model_instance, train_data, val_data, {'lr': lr, 'n_epochs': 5})
        max_val_acc = max(val_acc_history)
        accuracy_dict[lr] = max_val_acc # add best accuracy achieved for that learning rate

        if max_val_acc > best_accuracy:
            best_accuracy = max_val_acc
            best_lr = lr
   
    plt.figure(figsize=(8,6))
    plt.plot(accuracy_dict.keys(), accuracy_dict.values(), marker='o', linestyle='-')
    plt.xscale("log")
    plt.tick_params(axis='x', which='minor', length=0)
    plt.xticks(lr_list, labels=[str(lr) for lr in lr_list])
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy vs. Learning Rate ({model_name})")
    plt.savefig(f"lr_tuning_{model_name}.png")
    plt.close()

    print(f"Best model achieved {best_accuracy:.4f} validation accuracy with learning rate {best_lr}")

    return best_lr


### -------------------------- Group Normalization -------------------------- ###

class CNNgroupNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.n1 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.n2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.n3 = nn.GroupNorm(num_groups=16, num_channels=128)
        self.fc = nn.Linear(128*16*16, 100) 
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(100, 2) 

    def forward(self, X):
        c1 = self.n1(self.conv1(X))
        fm1 = F.max_pool2d(F.relu(c1), 2)
        c2 = self.n2(self.conv2(fm1))
        fm2 = F.max_pool2d(F.relu(c2), 2)
        c3 = self.n3(self.conv3(fm2))
        fm3 = F.max_pool2d(F.relu(c3), 2)
        fl = torch.flatten(fm3, start_dim=1)
        h1 = F.relu(self.fc(fl))
        h1 = self.dropout(h1)
        out = self.output(h1)
        return out


### -------------------------- Residual Connections ------------------------- ###

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()
   
    def forward(self, x):
        return self.activation(self.conv(x) + x) # skip connection

class CNNResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res3 = ResBlock(128)
        self.fc = nn.Linear(128*16*16, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(100, 2)
   
    def forward(self, X):
        c1 = self.conv1(X)
        fm1 = F.max_pool2d(F.relu(c1), 2)  
        c2 = self.conv2(fm1)
        fm2 = F.max_pool2d(F.relu(c2), 2)
        c3 = self.conv3(fm2)
        c3 = self.res3(c3)
        fm3 = F.max_pool2d(F.relu(c3), 2)
        fl = torch.flatten(fm3, start_dim=1)
        h1 = F.relu(self.fc(fl))
        h1 = self.dropout(h1)
        out = self.output(h1)
        return out


### ----------------- Group Normalization and Residual Connections ----------------- ###

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.activation = nn.ReLU()
   
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)) + x) 

class CNNResidualNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.n1 = nn.GroupNorm(num_groups=4, num_channels=32)
       
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.n2 = nn.GroupNorm(num_groups=8, num_channels=64)
       
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.n3 = nn.GroupNorm(num_groups=16, num_channels=128)
        self.res3 = ResBlock1(128)
       
        self.fc = nn.Linear(128*16*16, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(100, 2)
   
    def forward(self, X):
        c1 = self.n1(self.conv1(X))
        fm1 = F.max_pool2d(F.relu(c1), 2)
       
        c2 = self.n2(self.conv2(fm1))
        fm2 = F.max_pool2d(F.relu(c2), 2)
       
        c3 = self.n3(self.conv3(fm2))
        c3 = self.res3(c3)
        fm3 = F.max_pool2d(F.relu(c3), 2)
       
        fl = torch.flatten(fm3, start_dim=1)
        h1 = F.relu(self.fc(fl))
        h1 = self.dropout(h1)
        out = self.output(h1)
        return out
    

### -------------------------- Transfer Learning ------------------------- ###

def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for Xbatch, Ybatch in dataloader:
            feature_batch = model.features(Xbatch)
            feature_batch = feature_batch.view(feature_batch.size(0), -1) # flatten
            features.append(feature_batch.numpy())
            labels.append(Ybatch.numpy())

    return np.vstack(features), np.hstack(labels)


def save_vgg_features():
    weights_id = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    vggmodel = torchvision.models.vgg16(weights=weights_id)
    vggmodel.eval()
    vggtransforms = weights_id.transforms() # includes normalization

    pt_transform = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        vggtransforms
    ])
    vgg_train_dataset = ImageFolder('train', transform=pt_transform)
    vgg_val_dataset = ImageFolder('val', transform=vggtransforms)
    vgg_test_dataset = ImageFolder('test', transform=vggtransforms)

    vgg_train_loader = DataLoader(vgg_train_dataset, batch_size=8, shuffle=True)
    vgg_val_loader = DataLoader(vgg_val_dataset, batch_size=8, shuffle=False)
    vgg_test_loader = DataLoader(vgg_test_dataset, batch_size=8, shuffle=False)
 
    train_features, train_labels = extract_features(vggmodel, vgg_train_loader)
    val_features, val_labels = extract_features(vggmodel, vgg_val_loader)
    test_features, test_labels = extract_features(vggmodel, vgg_test_loader)

    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
    np.save('val_features.npy', val_features)
    np.save('val_labels.npy', val_labels)
    np.save('test_features.npy', test_features)
    np.save('test_labels.npy', test_labels)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # VGG16 feature size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes, MEL and NV
        )

    def forward(self, x):
        return self.classifier(x)


def create_mlp_dataloaders():
    train_features = np.load('train_features.npy')
    train_labels = np.load('train_labels.npy')
    val_features = np.load('val_features.npy')
    val_labels = np.load('val_labels.npy')
    test_features = np.load('test_features.npy')
    test_labels = np.load('test_labels.npy')

    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    batch_size = 32  # increase batch since we use features directly, not images

    mlp_train_dataset = TensorDataset(train_features, train_labels)
    mlp_val_dataset = TensorDataset(val_features, val_labels)
    mlp_test_dataset = TensorDataset(test_features, test_labels)

    mlp_train_loader = DataLoader(mlp_train_dataset, batch_size=batch_size, shuffle=True)
    mlp_val_loader = DataLoader(mlp_val_dataset, batch_size=batch_size, shuffle=False)
    mlp_test_loader = DataLoader(mlp_test_dataset, batch_size=batch_size, shuffle=False)

    return mlp_train_loader, mlp_val_loader, mlp_test_loader


### ------------------------ Feature Importance: Occlusion ------------------------ ###

def occlusion(dataloader, n_images=4):
    vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg_model.eval()

    # combine the vgg model with the classifier
    classifier = MLPClassifier() 
    full_model = nn.Sequential(
        vgg_model.features,
        nn.AdaptiveAvgPool2d((7, 7)),  # make sure feature maps are 7x7
        nn.Flatten(),
        classifier
    )

    occlusion = Occlusion(full_model) 

    batch = next(iter(dataloader))
    images, labels = batch
    images, labels = images[:n_images], labels[:n_images]

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))  
    fig.suptitle("Occlusion Analysis on VGG16 + Classifier", fontsize=16)

    for i in range(n_images):
        normalized_img = transform_normalize(images[i]).unsqueeze(0)  # add batch dimension

        output = full_model(normalized_img)
        pred_label_idx = output.argmax(dim=1).item()
        true_label_idx = labels[i].item()

        attributions_occ = occlusion.attribute(
            normalized_img,
            target=pred_label_idx,
            strides=(3, 4, 4),
            sliding_window_shapes=(3,15,15),
            baselines=0
        )

        orig_img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        attr_img = np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0))

        for j, method in enumerate(["original_image", "heat_map", "heat_map", "masked_image"]):
            ax = axes[i, j] if n_images > 1 else axes[j] 
            viz.visualize_image_attr(
                attr_img if method != "original_image" else None,
                orig_img,
                method=method,
                sign="all" if method == "original_image" else ("positive" if j == 1 else "negative"),
                show_colorbar=True,
                title=[f"Original", "Positive Attribution", "Negative Attribution", "Masked"][j],
                plt_fig_axis=(fig, ax) 
            )
        
    plt.tight_layout()
    plt.savefig("occlusion.png")
    plt.close()


### ---------------------------- Running the Models ---------------------------- ###

# Baseline model
torch.random.manual_seed(2) # for reproducibility
train_loader, val_loader, test_loader = create_dataloaders()
model_1 = BaselineCNN()
model_1_lr = tune_learningrate(BaselineCNN, train_loader, val_loader, model_name="Model_1")
val_acc_history_1, train_acc_history_1 = train_classifier(model_1, train_loader, val_loader, {'lr': model_1_lr, 'n_epochs': 20})
print(f"Best validation accuracy accieved for model 1: {max(val_acc_history_1)}")

# Group Normalization on Baseline model
torch.random.manual_seed(2)
model_2 = CNNgroupNormalization()
model_2_lr = tune_learningrate(CNNgroupNormalization, train_loader, val_loader, model_name="Model_2")
val_acc_history_2, train_acc_history_2 = train_classifier(model_2, train_loader, val_loader, {'lr': model_2_lr, 'n_epochs':20})
print(f"Best validation accuracy accieved for model 2: {max(val_acc_history_2)}")

# Residual Connections on Baseline model
torch.random.manual_seed(2)
model_3 = CNNResidual()
model_3_lr = tune_learningrate(CNNResidual, train_loader, val_loader, model_name="Model_3")
val_acc_history_3, train_acc_history_3 = train_classifier(model_3, train_loader, val_loader, {'lr': model_3_lr, 'n_epochs': 20}) 
print(f"Best validation accuracy accieved for model 3: {max(val_acc_history_3)}")

# Group Normalization and Residual Connections on Baseline model
torch.random.manual_seed(2)
model_4 = CNNResidualNorm()
model_4_lr = tune_learningrate(CNNResidualNorm, train_loader, val_loader, model_name="Model_4")
val_acc_history_4, train_acc_history_4 = train_classifier(model_4, train_loader, val_loader, {'lr': model_4_lr, 'n_epochs': 20}) 
print(f"Best validation accuracy accieved for model 4: {max(val_acc_history_4)}")

# Data Augmentation on Baseline model
torch.random.manual_seed(2)
aug_train_loader, aug_val_loader = create_aug_dataloaders()
val_acc_history_5, train_acc_history_5 = train_classifier(model_1, aug_train_loader, aug_val_loader, {'lr': model_1_lr, 'n_epochs': 20})
print(f"Best validation accuracy accieved for model 5: {max(val_acc_history_5)}")

# Transfer Learning
save_vgg_features() # run this only once
torch.random.manual_seed(2)
mlp_model = MLPClassifier() # the best model
mlp_train_loader, mlp_val_loader, mlp_test_loader = create_mlp_dataloaders() # includes test set 
model_6_lr = tune_learningrate(MLPClassifier, mlp_train_loader, mlp_val_loader, model_name="Model_6")
val_acc_history_6, train_acc_history_6 = train_classifier(mlp_model, mlp_train_loader, mlp_val_loader, {'lr': model_6_lr, 'n_epochs': 20})
print(f"Best validation accuracy accieved for model 6: {max(val_acc_history_6)}")


### --------------------------- Evaluating on the Test Set --------------------------- ###

# uncomment below for using the trained model, and comment out the training above
#mlp_model.load_state_dict(torch.load("MLPClassifier_best.pth"))
#print("MLP model successfully loaded")

# using MLP classifier with pre trained features
test_acc = predict_and_evaluate(mlp_model, mlp_test_loader)
print(f"Test accuracy for MLP model with pre-trained features: {test_acc}")
plot_misclassified_images_mlp(mlp_model, test_loader, mlp_test_loader, n_images=6)
occlusion(test_loader, n_images=4)