
import argparse
import os
import numpy as np
import math
import itertools
import sys
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import PIL
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import *
from SSIM_PIL import compare_ssim
from PIL import Image

from models import *
from datasets import *

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.makedirs("finaltraining", exist_ok=True)
os.makedirs("finalsaved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
#parser.add_argument("--dataset_name", type=str, default="NIH Chest X-Ray", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default= 5, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=30, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=24, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=50, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
#generator = nn.DataParallel(generator).to(device)
#discriminator = nn.DataParallel(discriminator).to(device)
feature_extractor = FeatureExtractor().to(device)
#feature_extractor = nn.DataParallel(feature_extractor).to(device)
# Set feature extractor to inference mode
feature_extractor.eval()
generator = generator.to(device)
# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("finalsaved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("finalsaved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), amsgrad = True)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), amsgrad = True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
dataloader = DataLoader(
    ImageDataset("HIGH_QUALITY_IMAGES", hr_shape = hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

from math import log10, sqrt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr,mse

def func1(genenerator,limit= 1000):
    psnr = []
    ssim = []
    mse = []
    #csv_path = "/DATA/vardhan1/SISR/ESRGAN/test_list.txt"
    eval_imgs_path = "low_quality_images/"  #"/DATA/vardhan1/SISR/ESRGAN/images/"
    #df = pd.read_csv(csv_path,header=None)
    output_dir = "images_RDB/"
    os.makedirs(output_dir, exist_ok=True)

    imgs = os.listdir(eval_imgs_path) #df[0].values
    for count,i in tqdm(enumerate(imgs)):
        img_path = eval_imgs_path + i
        if i.startswith('.'):
            continue  # Skip hidden files (e.g., .DS_Store)
        img_path = os.path.join(eval_imgs_path, i)
        if not os.path.isfile(img_path):
            continue
        try:
            ori = Image.open(img_path).convert("RGB").resize((1024, 1024))
        except (PIL.UnidentifiedImageError, OSError):
            continue
        lr=ori.resize((256,256))
        lr=Variable(transform(lr)).to(device).unsqueeze(0)
        with torch.no_grad():
            sr_image = denormalize(generator(lr)).cpu()[0]
        save_path = os.path.join(output_dir, i)
        save_image(sr_image, save_path)
        sr_image = Image.open(save_path).convert("RGB")
        #value = compare_ssim(sr_image,img_path)
        #image1 = Image.open(sr_image)
        image2 = Image.open(img_path).convert("RGB").resize((1024,1024))
        #print(image2.shape())
        #print(sr_image.shape())
        value = compare_ssim(sr_image , image2)


        #filename, ext = os.path.splitext(i)
        #sr_filename = f"{filename}_sr_{count:03d}.png"
        #save_image(sr_image, os.path.join(output_dir, sr_filename))
        #sr_image = Image.open(os.path.join(output_dir, "sr.png")).convert('RGB')
        ori = np.array(ori)
        sr_image = np.array(sr_image)
        evl=PSNR(ori,sr_image)
        #value = compare_ssim(ori,sr_image)
        mse.append(float(evl[1]))
        ssim.append(value)
        psnr.append(float(evl[0]))
        if count>limit:
            break

    return np.mean(psnr),np.mean(mse),np.mean(ssim)

df = pd.DataFrame(columns=["Epoch","PSNR","MSE","SSIM","Losses"])

# ----------
#  Training
# ----------
PSNR_fin=0
for epoch in range(opt.epoch, opt.n_epochs):
    amloss_G = AverageMeter()
    amloss_D = AverageMeter()
    amloss_content = AverageMeter()
    amloss_GAN = AverageMeter()
    amloss_pixel = AverageMeter()
    loop1 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, imgs in loop1:

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor)).to(device)
        imgs_hr = Variable(imgs["hr"].type(Tensor)).to(device)
        #imgs_lr2 = Variable(imgs["lr2"].type(Tensor)).to(device)
        #imgs_hr2 = Variable(imgs["hr2"].type(Tensor)).to(device)
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
       # print(imgs_lr)
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        amloss_pixel.update(loss_pixel.item(), imgs["lr"].size(0))

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        amloss_GAN.update(loss_GAN.item(), imgs["lr"].size(0))

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)
        amloss_content.update(loss_content.item(), imgs["lr"].size(0))
        # Part 2: For MSE_boundingbox
        #gen_hr2 = generator(imgs_lr2)
        #loss_pixel2 = criterion_pixel(gen_hr2, imgs_hr2)
        # Content loss
        #gen_features2 = feature_extractor(gen_hr2)
        #real_features2 = feature_extractor(imgs_hr2).detach()
        #loss_content2 = criterion_content(gen_features2, real_features2)


        # Total generator loss

        loss_G1 = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        #loss_G2 = loss_content2 + 0.1 * loss_pixel2
        #loss_G = loss_G1 + 0.2 * loss_G2
        # Total generator loss

        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        amloss_G.update(loss_G.item(), imgs["lr"].size(0))
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        amloss_D.update(loss_D.item(), imgs["lr"].size(0))

        loss_D.backward()
        optimizer_D.step()
        loop1.set_postfix(loss_G=amloss_G.avg, loss_D=amloss_D.avg, loss_content=amloss_content.avg, loss_pixel=amloss_pixel.avg, loss_GAN=amloss_GAN.avg)

        # --------------
        #  Log Progress
        # --------------

    psnr,mse,ssim = func1(generator)
    print("[Epoch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f] [PSNR: %f, MSE: %f , SSIM: %f]"
        % (epoch,
            opt.n_epochs,
            loss_D.item(),
            loss_G.item(),
            loss_content.item(),
            loss_GAN.item(),
            loss_pixel.item(),
            psnr,mse,ssim))
    data = [epoch,psnr,mse,ssim,[loss_D.item(),loss_G.item(),
            loss_content.item(),loss_GAN.item(),loss_pixel.item()]]
    df.loc[len(df.index)] = data
    if PSNR_fin<psnr:
        torch.save(generator.state_dict(), "finalsaved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "finalsaved_models/discriminator_%d.pth" %epoch)
        PSNR_fin = psnr
        print("===",psnr,"===")        
    df.to_csv("sq_RDB_ssim.csv")
print(df)



