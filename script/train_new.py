import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib
from functools import wraps
import pandas as pd
from .ssim_loss import SSIM
from .edge_loss import EdgeConvloss



# CNN, KAN, RES, UNET
def exp_train(model, opt_image, msi_image
          , device: int = 0
          , num_epochs=1000
          , interval_epochs=100
          , lr=0.001
          , save: str = None
          , loss_detail = False
          , criterion = nn.MSELoss()
          , exp_loss = 10
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
            loss = abs(exp_loss - criterion(output, msi_image))
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

# trian for loss finding
def train_lossiter(model, opt_image, msi_image
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          , exp_loss: int = 10
          , early_stop: int = 0.0005
          ):
    
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['loss'])

    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss = torch.abs(torch.tensor(exp_loss) - mse_loss(output, msi_image))
        if loss.item() < early_stop:
            print(f'Early stop at: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            break
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            losses.loc[epoch] = loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(losses.T) if losses.shape[0] > 0 else None


    return model, loss


# train for ssim
def train_ssim(model, opt_image, msi_image, ssim_ref
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          , exp_loss: int = 5
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['loss'])

    mse_loss = nn.MSELoss()
    ssim_loss = SSIM(data_range=1.0, channel=1)
    resize_transform = transforms.Resize(ssim_ref.shape[2:]
            , interpolation=transforms.InterpolationMode.BICUBIC)
    ssim_ref = ssim_ref.to(device)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss1 = torch.abs(exp_loss - mse_loss(output, msi_image))
        loss2 = 1 - ssim_loss(
            resize_transform(output)/torch.max(resize_transform(output))
            , ssim_ref/torch.max(ssim_ref)
            )
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.loc[epoch] = loss.item()
    print(losses.T)
    return model, losses


# 
def train_edge(model, opt_image, msi_image
          , edge_ref = None
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['loss'])

    mse_loss = nn.MSELoss()
    edge_loss = EdgeConvloss()
    if edge_ref is None:
        greyscale_transform = transforms.Grayscale(num_output_channels=1)
        edge_ref = greyscale_transform(opt_image).to(device)
    else:
        edge_ref = edge_ref.to(device)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss1 = mse_loss(output, msi_image)
        loss2 = edge_loss(output, edge_ref)
        loss = loss1*0.5 + loss2*5
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.loc[epoch] = loss.item()
    print(losses.T)
    return model, losses



# 
def train_edge_2(model, opt_image, msi_image
          , edge_ref = None
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['loss'])

    mse_loss = nn.MSELoss()
    edge_loss = EdgeConvloss()
    if edge_ref is None:
        greyscale_transform = transforms.Grayscale(num_output_channels=1)
        edge_ref = greyscale_transform(opt_image).to(device)
    else:
        edge_ref = edge_ref.to(device)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss1 = mse_loss(output, msi_image)
        loss2 = edge_loss(output, edge_ref)
        loss = loss1 + loss2*20
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            losses.loc[epoch] = loss.item()
    print(losses.T)
    return model, losses




#
def train_recordedge(model, opt_image, msi_image
          , edge_ref = None
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['mse', 'edge_diff'])

    mse_loss = nn.MSELoss()
    edge_loss = EdgeConvloss()
    if edge_ref is None:
        greyscale_transform = transforms.Grayscale(num_output_channels=1)
        edge_ref = greyscale_transform(opt_image).to(device)
    else:
        edge_ref = edge_ref.to(device)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss = mse_loss(output, msi_image)
        edge_diff = edge_loss(output, edge_ref)
        losses.loc[epoch] = [loss.item(), edge_diff.item()]
        loss.backward()
        optimizer.step()

        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, losses


# 
def train_step(model, opt_image, msi_image
          , edge_ref = None
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)
    losses = pd.DataFrame(columns=['mse', 'edge_diff'])

    mse_loss = nn.MSELoss()
    edge_loss = EdgeConvloss()
    if edge_ref is None:
        greyscale_transform = transforms.Grayscale(num_output_channels=1)
        edge_ref = greyscale_transform(opt_image).to(device)
    else:
        edge_ref = edge_ref.to(device)

    for epoch in tqdm(range(num_epochs//2)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss = mse_loss(output, msi_image)
        edge_diff = edge_loss(output, edge_ref)
        losses.loc[epoch] = [loss.item(), edge_diff.item()]
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, losses

# ssim+edge
def train_ssim_edge(model, opt_image, msi_image, ssim_ref
          , edge_ref = None
          , device: int = 0
          , num_epochs: int = 1000
          , interval_epochs: int = 100
          , lr: float = 0.001
          ):
    
    device = torch.device(f'cuda:{device}')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    opt_image, msi_image = opt_image.to(device), msi_image.to(device)

    mse_loss = nn.MSELoss()

    ssim_loss = SSIM(data_range=1.0, channel=1)
    resize_transform = transforms.Resize(ssim_ref.shape[2:]
            , interpolation=transforms.InterpolationMode.BICUBIC)
    ssim_ref = ssim_ref.to(device)

    edge_loss = EdgeConvloss()
    if edge_ref is None:
        greyscale_transform = transforms.Grayscale(num_output_channels=1)
        edge_ref = greyscale_transform(opt_image).to(device)
    else:
        edge_ref = edge_ref.to(device)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(opt_image)
        loss1 = 1 - ssim_loss(
            resize_transform(output)/torch.max(resize_transform(output))
            , ssim_ref/torch.max(ssim_ref)
            )
        loss2 = edge_loss(output, edge_ref)
        loss3 = mse_loss(output, msi_image)
        loss = loss1 + loss2 + loss3*0.2
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % interval_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model