import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
from functools import wraps
import pandas as pd



# CNN, KAN, RES, UNET
def train(model, opt_image, msi_image
          , device: int = 0
          , num_epochs=1000
          , interval_epochs=100
          , lr=0.001
          , save: str = None
          , loss_detail = False
          , criterion = nn.MSELoss()
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['loss'])

    if loss_detail == False:
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            output = model(opt_image)
            loss = criterion(output, msi_image)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % interval_epochs == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                losses.loc[epoch] = loss.item()
                if save != None:
                    torch.save(model.state_dict(), save[0]+f'3kx3k-{epoch}.pth')

                    
    if loss_detail == True:
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            output = model(opt_image)
            loss = criterion(output, msi_image)
            loss.backward()
            optimizer.step()
            losses.loc[epoch] = loss.item()
            
            if (epoch + 1) % interval_epochs == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                if save != None:
                    torch.save(model.state_dict(), save[0]+f'3kx3k-{epoch}.pth')

                    
    return model, losses
