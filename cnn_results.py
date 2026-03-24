import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from train_cnn import EmotionCNN, device, val_loader
epoch_num = 50

def visualize_saved_model(model_path, val_loader, device):
    # 1. Re-initialize the model architecture
    model = EmotionCNN(num_emotions=4).to(device)
    
    # 2. Load the saved "brain" (weights)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Run a quick evaluation pass
    print("Testing saved model on validation set...")
    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            outputs = model(specs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 4. Plot the Confusion Matrix
    emotions = ["Happy", "Angry", "Sad", "Neutral"]
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion Matrix: {model_path}")
    plt.savefig(f"{epoch_num}_epochs_confusion_matrix.png")
    plt.show()

visualize_saved_model(f"{epoch_num}_epochs_emotion_cnn.pth", val_loader, device)