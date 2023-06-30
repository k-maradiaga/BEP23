import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import argparse
from progress.bar import IncrementalBar
from PIL import Image
import glob
import tqdm
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from piq import psnr, SSIMLoss

parser = argparse.ArgumentParser(prog = 'top', description='Train GAN')
parser.add_argument("--modelname", type=str, default='pix2pix', help="['pix2pix', 'unetr2d']")
parser.add_argument("--db", type=str, default='all', help="['10db', '20db', '30db', '40db', '50db', 'all']")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="all", help="Name of the dataset: ['drive', 'nne', 'all']")
parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
db_list = [args.db]
EPOCHS = args.epochs
modelname = args.modelname
datasetname_list = [args.dataset]

if datasetname_list[0] == 'all':
    print('All datasets')
    datasetname_list = ['drive', 'nne']

for datasetname in datasetname_list:

    if db_list[0] == 'all':
        print("All noise levels")
        db_list = ['10db', '20db', '30db', '40db', '50db']

    for db in db_list:

        # check file extension
        if datasetname == 'nne':
            datatype = 'png'
            EPOCHS = 10
        else:
            datatype = 'jpg'

        device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Print info
        print(db, datasetname, device)

        # Weights init
        def initialize_weights(layer):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)
            return None

        class GeneratorLoss(nn.Module):
            def __init__(self, alpha=100):
                super().__init__()
                self.alpha=alpha
                self.bce=nn.BCEWithLogitsLoss()
                self.l1=nn.L1Loss()
                self.ssiml = SSIMLoss()

    
            def forward(self, fake, real, fake_pred):
                fake_target = torch.ones_like(fake_pred)
                l1tje = self.alpha* self.l1(fake,real)
                ssimtje = self.alpha* self.ssiml((fake + 1)/ 2, (real +1) / 2)
                loss = self.bce(fake_pred, fake_target) + ssimtje
                return loss
            
            
        class DiscriminatorLoss(nn.Module):
            def __init__(self,):
                super().__init__()
                self.loss_fn = nn.BCEWithLogitsLoss()
                
            def forward(self, fake_pred, real_pred):
                fake_target = torch.zeros_like(fake_pred)
                real_target = torch.ones_like(real_pred)
                fake_loss = self.loss_fn(fake_pred, fake_target)
                real_loss = self.loss_fn(real_pred, real_target)
                loss = (fake_loss + real_loss)/2
                return loss

        # DATASET
        class Dataset(torch.utils.data.Dataset):
                def __init__(self, input_dir, label_dir, transform = None, mode='train'):
                    super(Dataset, self).__init__()
                    self.label_dir = label_dir
                    self.file_list = input_dir
                    self.transform = transform
            
                def __len__(self):
                    return len(self.file_list)
            
                def collate(self, batch):
                    return [torch.cat(v) for v in zip(*batch)]
            
                def __getitem__(self, index):
                    img_path = self.file_list[index]
                    img = Image.open(img_path).convert('RGB')
                    img_out =  self.transform(img)
            
                    label_path = self.label_dir[index]
                    label = Image.open(label_path).convert('RGB')
                    label_out =  self.transform(label)
            
                    return img_out, label_out
                
        train_target_list = []
        for filename in glob.glob('thesis_data/thesis_data/{0}/train/ground_truth/*.{1}'.format(datasetname, datatype)):
            train_target_list.append(filename)

        train_target_list = sorted(train_target_list, key=lambda fname: int(fname.split('.')[0][-5:]))

        train_input_list = []
        
        for filename in glob.glob('thesis_data/thesis_data/{0}/train/{1}/*.{2}'.format(datasetname, db, datatype)):
            train_input_list.append(filename)

        train_input_list = sorted(train_input_list, key=lambda fname: int(fname.split('.')[0][-5:]))


        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])


        val_input_list = train_input_list[-int(len(train_input_list) * .20):]
        val_target_list = train_target_list[-int(len(train_target_list) * .20):]

        train_input_list = train_input_list[0: int(len(train_input_list) * 0.8)]
        train_target_list = train_target_list[0: int(len(train_target_list) * 0.8)]

        print("validation input:", len(val_input_list))
        print("validation target:", len(val_target_list))

        val_dataset = Dataset(val_input_list, val_target_list, transform, 'val')

        train_dataset = Dataset(train_input_list, train_target_list, transform,'train')

        test_target_list = []
        for filename in glob.glob('experimental-data2/{0}/ground_truth/*.png'.format(datasetname, datatype)):
            test_target_list.append(filename)

        test_target_list = sorted(test_target_list, key=lambda fname: int(fname.split('.')[0][-5:]))

        test_input_list = []
        for filename in glob.glob('experimental-data2/{0}/test/{1}/*.png'.format(datasetname, db, datatype)):
            test_input_list.append(filename)

        test_input_list = sorted(test_input_list, key=lambda fname: int(fname.split('.')[0][-5:]))


        test_dataset = Dataset(test_input_list, test_target_list, transform,'test')

        print("train:",len(train_dataset))
        print("test:",len(test_dataset))

        train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        #######################################################################
        def flatten(l):
            return [item for sublist in l for item in sublist]

        import statistics
        def test_metrics(generator, dl):
            ssims_test = []
            psnrs_test = []
            x_list = []
            y_list = []
            
            for i, (inputs, targets) in enumerate(dl):
                inputs = inputs.to(device)
                targets = targets.to(device)
                y_pred = generator(inputs)
               
               
                # CALCULATE ssim and psnr
                X = (y_pred.cpu().detach() + 1) / 2  # [-1, 1] => [0, 1]
                Y = (targets.cpu().detach() + 1) / 2
                
                x_list.append(X)
                y_list.append(Y)
                
                ssim_t = ssim(Y, X, data_range=1, size_average = False).cpu().detach()
                ssims_test.append(np.array(ssim_t.tolist()))
                psnr_t = psnr(Y, X, data_range=1, reduction='none' )
                psnrs_test.append(np.array(psnr_t.tolist()))

            ssims_test = flatten(ssims_test)
            psnrs_test = flatten(psnrs_test)

            print("SSIM mean:", statistics.mean(np.array(ssims_test)))
            print("PSNR mean:", statistics.mean(np.array(psnrs_test)))

            mean_ssim = statistics.mean(np.array(ssims_test))
            return ssims_test, psnrs_test, mean_ssim, x_list, y_list


        #################################################################
        ## GENERATORS
        ###################################################################
        if modelname == 'pix2pix':
            class EncoderBlock(nn.Module):
                """Encoder block"""
                def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
                    super().__init__()
                    self.lrelu = nn.LeakyReLU(0.2, inplace=True)
                    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
                    
                    self.bn=None
                    if norm:
                        self.bn = nn.BatchNorm2d(outplanes)
                    
                def forward(self, x):
                    fx = self.lrelu(x)
                    fx = self.conv(fx)
                    
                    if self.bn is not None:
                        fx = self.bn(fx)
                        
                    return fx

                
            class DecoderBlock(nn.Module):
                """Decoder block"""
                def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
                    super().__init__()
                    self.relu = nn.ReLU(inplace=True)
                    self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
                    self.bn = nn.BatchNorm2d(outplanes)       
                    
                    self.dropout=None
                    if dropout:
                        self.dropout = nn.Dropout2d(p=0.5, inplace=True)
                        
                def forward(self, x):
                    fx = self.relu(x)
                    fx = self.deconv(fx)
                    fx = self.bn(fx)

                    if self.dropout is not None:
                        fx = self.dropout(fx)
                        
                    return fx    
                
            class UnetGenerator(nn.Module):
                """Unet-like Encoder-Decoder model"""
                def __init__(self,):
                    super().__init__()
                    
                    self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
                    self.encoder2 = EncoderBlock(64, 128)
                    self.encoder3 = EncoderBlock(128, 256)
                    self.encoder4 = EncoderBlock(256, 512)
                    self.encoder5 = EncoderBlock(512, 512)
                    self.encoder6 = EncoderBlock(512, 512)
                    self.encoder7 = EncoderBlock(512, 512)
                    self.encoder8 = EncoderBlock(512, 512, norm=False)
                    
                    self.decoder8 = DecoderBlock(512, 512, dropout=True)
                    self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
                    self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
                    self.decoder5 = DecoderBlock(2*512, 512)
                    self.decoder4 = DecoderBlock(2*512, 256)
                    self.decoder3 = DecoderBlock(2*256, 128)
                    self.decoder2 = DecoderBlock(2*128, 64)
                    self.decoder1 = nn.ConvTranspose2d(2*64, 3, kernel_size=4, stride=2, padding=1)
                    
                def forward(self, x):
                    # encoder forward
                    e1 = self.encoder1(x)
                    e2 = self.encoder2(e1)
                    e3 = self.encoder3(e2)
                    e4 = self.encoder4(e3)
                    e5 = self.encoder5(e4)
                    e6 = self.encoder6(e5)
                    e7 = self.encoder7(e6)
                    e8 = self.encoder8(e7)
                    # decoder forward + skip connections
                    d8 = self.decoder8(e8)
                    d8 = torch.cat([d8, e7], dim=1)
                    d7 = self.decoder7(d8)
                    d7 = torch.cat([d7, e6], dim=1)
                    d6 = self.decoder6(d7)
                    d6 = torch.cat([d6, e5], dim=1)
                    d5 = self.decoder5(d6)
                    d5 = torch.cat([d5, e4], dim=1)
                    d4 = self.decoder4(d5)
                    d4 = torch.cat([d4, e3], dim=1)
                    d3 = self.decoder3(d4)
                    d3 = torch.cat([d3, e2], dim=1)
                    d2 = F.relu(self.decoder2(d3))
                    d2 = torch.cat([d2, e1], dim=1)
                    d1 = self.decoder1(d2)
                    
                    return torch.tanh(d1)
            
            generator = UnetGenerator().to(device)

        elif modelname == 'unetr2d':
            from typing import Tuple, Union
            from monai.networks.blocks.dynunet_block import UnetOutBlock
            from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
            from monai.networks.nets import ViT

            class UNETR2D(nn.Module):
                """
                UNETR based on: "Hatamizadeh et al.,
                UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
                """
            
                def __init__(
                    self,
                    in_channels: int,
                    out_channels: int,
                    img_size: Tuple[int, int],
                    feature_size: int = 16,
                    hidden_size: int = 768,
                    mlp_dim: int = 3072,
                    num_heads: int = 12,
                    pos_embed: str = "perceptron",
                    norm_name: Union[Tuple, str] = "instance",
                    conv_block: bool = False,
                    res_block: bool = True,
                    dropout_rate: float = 0.0,
                ) -> None:
            
                    super().__init__()
            
                    if not (0 <= dropout_rate <= 1):
                        raise AssertionError("dropout_rate should be between 0 and 1.")
            
                    if hidden_size % num_heads != 0:
                        raise AssertionError("hidden size should be divisible by num_heads.")
            
                    if pos_embed not in ["conv", "perceptron"]:
                        raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
            
                    self.num_layers = 12
                    self.patch_size = (16, 16)
                    self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
                    self.hidden_size = hidden_size
                    self.classification = False
                    self.vit = ViT(
                        in_channels=in_channels,
                        img_size=img_size,
                        patch_size=self.patch_size,
                        hidden_size=hidden_size,
                        mlp_dim=mlp_dim,
                        num_layers=self.num_layers,
                        num_heads=num_heads,
                        pos_embed=pos_embed,
                        classification=self.classification,
                        dropout_rate=dropout_rate,
                        spatial_dims=2
                    )
                    self.encoder1 = UnetrBasicBlock(
                        spatial_dims=2,
                        in_channels=in_channels,
                        out_channels=feature_size,
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                        res_block=res_block,
                    )
                    self.encoder2 = UnetrPrUpBlock(
                        spatial_dims=2,
                        in_channels=hidden_size,
                        out_channels=feature_size * 2,
                        num_layer=2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
                    )
                    self.encoder3 = UnetrPrUpBlock(
                        spatial_dims=2,
                        in_channels=hidden_size,
                        out_channels=feature_size * 4,
                        num_layer=1,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
                    )
                    self.encoder4 = UnetrPrUpBlock(
                        spatial_dims=2,
                        in_channels=hidden_size,
                        out_channels=feature_size * 8,
                        num_layer=0,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
                    )
                    self.decoder5 = UnetrUpBlock(
                        spatial_dims=2,
                        in_channels=hidden_size,
                        out_channels=feature_size * 8,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                    )
                    self.decoder4 = UnetrUpBlock(
                        spatial_dims=2,
                        in_channels=feature_size * 8,
                        out_channels=feature_size * 4,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                    )
                    self.decoder3 = UnetrUpBlock(
                        spatial_dims=2,
                        in_channels=feature_size * 4,
                        out_channels=feature_size * 2,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                    )
                    self.decoder2 = UnetrUpBlock(
                        spatial_dims=2,
                        in_channels=feature_size * 2,
                        out_channels=feature_size,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                    )
                    self.out = nn.Sequential(UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels), nn.Tanh())  # type: ignore
            
                def proj_feat(self, x, hidden_size, feat_size):
                    x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    return x
            
                def forward(self, x_in):
                    x, hidden_states_out = self.vit(x_in)
                    enc1 = self.encoder1(x_in)
                    x2 = hidden_states_out[3]
                    enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
                    x3 = hidden_states_out[6]
                    enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
                    x4 = hidden_states_out[9]
                    enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
                    dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
                    dec3 = self.decoder5(dec4, enc4)
                    dec2 = self.decoder4(dec3, enc3)
                    dec1 = self.decoder3(dec2, enc2)
                    out = self.decoder2(dec1, enc1)
                    logits = self.out(out)
                    return logits
                
            generator = UNETR2D(in_channels=3, out_channels=3, img_size=(256, 256), feature_size=32, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="batch", conv_block=True, res_block=True, dropout_rate=0.5).to(device)
        else:
            print("no model")


        ##################################################################################################################################
        ## DISCRIMINATOR
        ##################################################################################################################################

        class BasicBlock(nn.Module):
            """Basic block"""
            def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
                super().__init__()
                self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
                self.isn = None
                if norm:
                    self.isn = nn.InstanceNorm2d(outplanes)
                self.lrelu = nn.LeakyReLU(0.2, inplace=True)
                
            def forward(self, x):
                fx = self.conv(x)
                
                if self.isn is not None:
                    fx = self.isn(fx)
                    
                fx = self.lrelu(fx)
                return fx
            
        class ConditionalDiscriminator(nn.Module):
            """Conditional Discriminator"""
            def __init__(self,):
                super().__init__()
                self.block1 = BasicBlock(6, 64, norm=False)
                self.block2 = BasicBlock(64, 128)
                self.block3 = BasicBlock(128, 256)
                self.block4 = BasicBlock(256, 512)
                self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
                
            def forward(self, x, cond):
                x = torch.cat([x, cond], dim=1)
                # blocks forward
                fx = self.block1(x)
                fx = self.block2(fx)
                fx = self.block3(fx)
                fx = self.block4(fx)
                fx = self.block5(fx)
                
                return fx
        #################################################################################################################

        # models
        print('Defining models!')
        print(generator)
        discriminator = ConditionalDiscriminator().to(device)

        # optimizers
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # loss functions
        g_criterion = GeneratorLoss(alpha=100)
        d_criterion = DiscriminatorLoss()

        # Log results
        best_test_ssim = []
        best_test_psnr = []
        x_final_list = []
        y_final_list = []
        val_ssim_final = []
        val_psnr_final = []
        train_ssim_final = []
        train_psnr_final = []

        # training loop
        best_ssim_v = 0
        bar_i = int(len(train_dataset) / BATCH_SIZE)
        print('Start of training process!')
        for epoch in range(EPOCHS):
            ge_loss=0.
            de_loss=0.
            train_ssim = []
            train_psnr = []
            val_ssim = []
            val_psnr = []
            start = time.time()
            bar = IncrementalBar(f'[Epoch {epoch+1}/{EPOCHS}]', max=bar_i)
            for x, real in train_dl:
                x = x.to(device)
                real = real.to(device)

                # Generator`s loss
                fake = generator(x)
                fake_pred = discriminator(fake, x)
                g_loss = g_criterion(fake, real, fake_pred)

                # Discriminator`s loss
                fake = generator(x).detach()
                fake_pred = discriminator(fake, x)
                real_pred = discriminator(real, x)
                d_loss = d_criterion(fake_pred, real_pred)

                # Generator`s params update
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # Discriminator`s params update
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Calculate training ssim and psnr
                X = (fake.cpu().detach() + 1) / 2
                Y = (real.cpu().detach() + 1) / 2
                ssim_t = ssim(X, Y, data_range=1, size_average=True)
                train_ssim.append(ssim_t)
                psnr_t = psnr(X, Y, data_range=1, reduction='mean')
                train_psnr.append(psnr_t)

                # add batch losses
                ge_loss += g_loss.item()
                de_loss += d_loss.item()
                bar.next()
            bar.finish()  

            # obtain per epoch losses
            g_loss = ge_loss/len(test_dl)
            d_loss = de_loss/len(test_dl)

            ssim_epoch = np.mean(train_ssim)
            psnr_epoch = np.mean(train_psnr)
            train_ssim_final.append(ssim_epoch)
            train_psnr_final.append(psnr_epoch)

            # count timeframe
            end = time.time()
            tm = (end - start)
            print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, EPOCHS, g_loss, d_loss, tm))
            
            # Save val metrics each epoch
            print("Validation Score:")
            v_ssim, v_psnr, v_ssim, _, _ = test_metrics(generator, val_dl)
            val_ssim.append(v_ssim)
            val_psnr.append(v_psnr)

            val_ssim_final.append(np.mean(np.array(val_ssim)))
            val_psnr_final.append(np.mean(np.array(val_psnr)))

            if v_ssim > best_ssim_v:
                print("Test score:")
                t_ssim, t_psnr, _, x_final, y_final = test_metrics(generator, test_dl)
                best_test_ssim = t_ssim
                best_test_psnr = t_psnr
                x_final_list = x_final
                y_final_list = y_final
                print('Better score')
                print()
                best_ssim_v = v_ssim
        print('End of training process!')
        print('Saving results ...')

        ssim_train_name = 'final/train_ssim_{0}_{1}.csv'.format(db + str(BATCH_SIZE), modelname + datasetname)
        ssim_test_name = 'final/test_ssim_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)
        psnr_test_name = 'final/test_psnr_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)
        psnr_train_name = 'final/train_psnr_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)

        ssim_val_name = 'final/val_ssim_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)
        psnr_val_name = 'final/val_psnr_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)
        g_loss_name = 'final/g_loss_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)
        d_loss_name = 'final/d_loss_{0}_{1}.csv'.format(db+ str(BATCH_SIZE), modelname+datasetname)

        test_ssim = best_test_ssim
        test_psnr = best_test_psnr

        x_final_list = flatten(x_final_list)
        y_final_list = flatten(y_final_list)
        train_ssim = torch.stack(train_ssim).tolist()
        train_psnr = torch.stack(train_psnr).tolist()

        np.savetxt(ssim_train_name, np.array(train_ssim_final), delimiter=",", fmt='%s')
        np.savetxt(psnr_train_name, np.array(train_psnr_final), delimiter=",", fmt='%s')

        np.savetxt(ssim_test_name, np.array(test_ssim), delimiter=",", fmt='%s')
        np.savetxt(psnr_test_name, np.array(test_psnr), delimiter=",", fmt='%s')


        np.savetxt(ssim_val_name, np.array(val_ssim_final), delimiter=",", fmt='%s')
        np.savetxt(psnr_val_name, np.array(val_psnr_final), delimiter=",", fmt='%s')

        torch.save(x_final_list, 'finalimage4/x_list_{0}_{1}.pt'.format(db + str(BATCH_SIZE), datasetname + '_' + modelname))
        torch.save(y_final_list, 'finalimage4/y_list_{0}_{1}.pt'.format(db + str(BATCH_SIZE), datasetname + '_' + modelname))

        print('Succesfully saved!')
        #np.savetxt(g_loss_name, np.array(final_g), delimiter=",", fmt='%s')
        #np.savetxt(d_loss_name, np.array(final_d), delimiter=",", fmt='%s')
