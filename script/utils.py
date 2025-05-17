#####################################################################
##                      Image functions                     #########
#####################################################################
import numpy as np
import torch

# image generation with trained model
def genimg(model, x):

    match x.shape:
        case (1, 3, i, j):
            pass
        case (3, i, j):
            x = x.reshape(1, 3, i, j)
        case (i, j, 3):
            x = np.swapaxes(x, -1, 1)
            x = x.reshape(1, 3, i, j)
        case (i, j):
            x = x.reshape(1, 1, i, j)

    match x:
        case np.ndarray():
            x = torch.tensor(x, dtype=torch.float32)
        case _:
            pass

    with torch.no_grad():
        model.to(torch.device('cpu'))
        y_new = model(x)
        y_new = y_new.squeeze().numpy()
    return y_new

# crop array by [up, down, left, right]
def crop_2d(x, shape: list[int], hw_index: int = 0):
    up, down, left, right = shape
    if hw_index == -1:
        return x[:, up:down, left:right].copy()
    elif hw_index == 0:
        return x[up:down, left:right].copy()


# nomralization by whole array
def MaxMinNormalization(x, range_: tuple = (0, 1)):
    ma = np.float32(range_[1])
    mi = np.float32(range_[0])
    cha = ma - mi
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min)*cha / (Max - Min)
    x = x + mi
    return x



#####################################################################
##                      calculation functions               #########
#####################################################################
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image


# entropy calculation
def calculate_entropy(image):
    hist = np.histogram(image, bins=256)[0] / image.size
    return -np.sum([p * np.log2(p) for p in hist if p > 0])




# ssim and psnr calculation
def calculate_ssim_psnr(img1_arr, img2_arr, normalize:tuple = (0, 1)):
    h1, w1 = img1_arr.shape
    h2, w2 = img2_arr.shape
    min_h, min_w = min(h1, h2), min(w1, w2)
    
    img1_resized = np.array(Image.fromarray(img1_arr).resize((min_w, min_h)))
    img2_resized = np.array(Image.fromarray(img2_arr).resize((min_w, min_h)))

    img1_resized = MaxMinNormalization(img1_resized, range_=normalize)
    img2_resized = MaxMinNormalization(img2_resized, range_=normalize)
    
    data_range = max(normalize) - min(normalize)
    ssim_value = ssim(img1_resized, img2_resized, 
                     data_range=data_range)
    psnr_value = psnr(img1_resized, img2_resized, 
                     data_range=data_range)
    
    return ssim_value, psnr_value


#####################################################################
########                plotting functions               ############
#####################################################################
import matplotlib.pyplot as plt

def show_image(image
               , figsize = None
               , cmap = None
               , axis = False
               ):

    if figsize:
        plt.figure(figsize=figsize)

    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

    if not axis:
        plt.axis('off')
    plt.show()


def show_images(
        images: list[np.ndarray]
        , layout: tuple[int, int] = None
        , cmap: str = None
        , titles: list[str] = None
        , figsize = None
        , **kwargs  # additional arguments for plt.subplots

    ):
    """
    Show multiple images in a grid.
    """
    # planing layout
    rows, cols = layout if layout else (1, len(images))

    # calculate figsize
    if type(figsize) is str:
        if len(images[0].shape) == 3:
            h, w, _ = images[0].shape
        elif len(images[0].shape) == 2:
            h, w = images[0].shape
        else:
            raise ValueError("Invalid image shape")
        figsize = (w * cols * 3, h * rows * 3)
    
    # create subplots
    _, axes = plt.subplots(rows, cols
                           , subplot_kw = {"xticks":[],"yticks":[]}
                           , figsize=figsize
                           , **kwargs
                           )
    
    for i, ax in enumerate(axes.flat if type(axes) is np.ndarray else [axes]):
        ax.imshow(images[i], cmap=cmap)
        ax.axis('off')
        if titles:
            ax.set_title('{}'.format(titles[i])
                         , fontsize = 20
                         , color = 'black')
    
    plt.tight_layout()
    plt.show()