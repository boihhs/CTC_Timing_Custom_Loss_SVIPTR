"""
Testing for the Model
Written by Leo Benaharon
"""

from SVIIPTR import SVIPTR
from dataload import ImageTextDataset
from dataloadRealWorld import ImageTextDataset2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataload import vocab
from torch.utils.data import DataLoader, Subset
from dataload import collate_fn
import torch.optim as optim
from torch.optim import AdamW
from CTCLoss import CTCLoss
from dataloadSVT import SVTDataset
from dataload import vocab_reverse
from autocorrect import Speller
import Levenshtein




if __name__ == '__main__':



    # Load data
    dataset = ImageTextDataset2('RealWorldDataSets/ICARtest/ground_truth.txt', 'RealWorldDataSets/ICARtest/img', vocab)
    #dataset = ImageTextDataset(annotations_file='dataset/text_instances_test.pkl', img_dir= 'dataset/test', vocab=vocab)
    #dataset = SVTDataset(xml_path='svt1/test.xml', image_folder='svt1/img', vocab=vocab)
    
   


    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, collate_fn=collate_fn)


    # Load model
    model = SVIPTR()
    model.load_state_dict(torch.load("model_weights_final_1.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    # Count how many correct/wrong
    count = 0
    count_correct = 0
    count_correct_2 = 0

    for batch in test_loader: 
        #images, Target_lengths, Targets, Input_lengths = batch
        #images, Target_lengths, Targets, Input_lengths = images.to(device), Target_lengths.to(device), Targets.to(device), Input_lengths.to(device)
        images, lengths, labels = batch
        images, lengths, labels = images.to(device), lengths.to(device), labels.to(device)

        # Get the highest probablity for each timestep in the batch
        model_out = model(images)
        model_out = torch.softmax(model_out, 2)
        model_out = torch.argmax(model_out, dim=2)

        # Loop through the batch and see if it is correct
        for i in range(model_out.shape[0]):
            
            out = model_out[i, :] # Get the respective output
            mask_out = torch.ones_like(out, dtype=torch.bool) # Generate a mask
            mask_out[1:] = out[1:] != out[:-1] # Check if characters repeat
            out_removed_repeats = out[mask_out] # Remove the repeats
            out_removed_repeats_and_zeros = out_removed_repeats[out_removed_repeats != 0] # Remove the zeros
            out_str_list = [vocab_reverse[char.item()] for char in out_removed_repeats_and_zeros] # Convert it to the vocab
            out_str = ''.join(j for j in out_str_list).lower() # Convert it to a string
          

            # Same thing for the labels
            label = labels[i, :]
            mask_label = torch.ones_like(label, dtype=torch.bool)
            mask_label[1:] = label[1:] != label[:-1]
            label_removed_repeats = label[mask_label]
            label_removed_repeats_and_zeros = label_removed_repeats[label_removed_repeats != 0]
            list_str_list = [vocab_reverse[char.item()] for char in label_removed_repeats_and_zeros]
            label_str = ''.join(j for j in list_str_list).lower()

            count += 1
            if len(label_str) == 0:
                if out_str == '':
                    count_correct += 1.0  # both empty → perfect match
                else:
                    count_correct += 0.0  # label is empty, prediction is not
            else:
                dist = Levenshtein.distance(out_str.lower(), label_str.lower())
                count_correct += 1 - dist / len(label_str)

            if out_str.lower() == label_str.lower():
                count_correct_2 += 1

            
            # Write to a file to see the outputs
            with open('predicted.txt', 'a') as f:
                f.write(out_str + '\n')

            with open('acutal.txt', 'a') as f:
                f.write(label_str + '\n')

    # Print results
    print("Count Correct: " + str(count_correct))
    print("Count: " + str(count))
    print("% Correct: " + str(count_correct / (count)))
    print("% Correct 2: " + str(count_correct_2 / (count)))




        
