"""
Training for the Model. Some of the commented things throughout othe code including other files are for the built in ctc loss.
This is good as I might have not used it correctly
Written by Leo Benaharon
"""

from SVIIPTR import SVIPTR
from dataload import ImageTextDataset
from dataloadRealWorld import ImageTextDataset2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloadRealWorld import vocab
from torch.utils.data import DataLoader, Subset
from dataloadRealWorld import collate_fn
import torch.optim as optim
from torch.optim import AdamW
from CTCLossTimingIsKey import CTCLoss
from dataloadSVT import SVTDataset
from transformers import get_cosine_schedule_with_warmup




if __name__ == '__main__':
    # Load data
    dataset = ImageTextDataset2('RealWorldDataSets/fullDatasetRealWorld/ground_truth.txt', 'RealWorldDataSets/fullDatasetRealWorld/img', vocab)
    #dataset = ImageTextDataset(annotations_file='dataset/text_instances_train.pkl', img_dir= 'dataset/train', vocab=vocab)
    #dataset = SVTDataset(xml_path='svt1/train.xml', image_folder='svt1/img', vocab=vocab)
    
   


    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, collate_fn=collate_fn)


    # Load model
    model = SVIPTR()
    model.load_state_dict(torch.load("model_weights_final_1.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (trainable + non-trainable): {total_params:,}")

    # Load loss
    criterion = CTCLoss(T=24).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    num_epochs = 20

    # Steps per epoch = 70000 // batch_size
    batch_size = 32
    steps_per_epoch = len(train_loader)  # ≈ 1093
    print(steps_per_epoch)
    total_steps = num_epochs * steps_per_epoch     # for 15 epochs of pretraining
    warmup_steps = int(0.05 * total_steps) # 5% warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        # Run the model training
        model.train()
        running_loss = 0.0

        for batch in train_loader: 
            #images, Target_lengths, Targets, Input_lengths = batch
            images, lengths, labels = batch
            images, lengths, labels = images.to(device), lengths.to(device), labels.to(device)
          
            optimizer.zero_grad() # Zero grad to not keep weights from previous batch
            model_out = model(images)
            loss = criterion(model_out, labels, lengths) 
     
            loss.backward()  # Backpropagation
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), "model_weights_final_2_2.pth")
