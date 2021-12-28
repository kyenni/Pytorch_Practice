import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from data_loader import Dataset
from network import UNet


lr=1e-3
batch_size=4
num_epoch=100

data_train_dir='./drive/MyDrive/Colab Notebooks/archive (4)/train'
data_val_dir='./drive/MyDrive/Colab Notebooks/archive (4)/val'
data_test_dir='./drive/MyDrive/Colab Notebooks/archive (4)/test'
ckpt_dir='./drive/MyDrive/Colab Notebooks/archive (4)/checkpoint'
log_dir='./drive/MyDrive/Colab Notebooks/archive (4)/log'
result_dir='./drive/MyDrive/Colab Notebooks/archive (4)/result_test'

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir,'png'))
    os.makedirs(os.path.join(result_dir,'numpy'))

transform_=transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.3),transforms.ToTensor(),transforms.Resize((512,512)),transforms.Normalize((0.5), (0.5))])
transform_t=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.3),
                                transforms.ToTensor(),
                                transforms.Resize((512,512))])

dataset_test=Dataset(data_test_dir,transform=transform_,target_transform=transform_t)
loader_test=DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=4)

net=UNet().to(device)

fn_loss=nn.BCEWithLogitsLoss().to(device)

optim=torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.999))

num_data_test=len(dataset_test)


num_batch_test=np.ceil(num_data_test/batch_size)


fn_tonumpy=lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm=lambda x,mean,std : (x*std)+mean
fn_class=lambda x : 1.0*(x>0.5)


def save(ckpt_dir,net,optim,epoch):
    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
                "./%s/model_epoch%d.pth"%(ckpt_dir,epoch))

def load(ckpt_dir,net,optim):
    ckpt_lst=os.listdir(ckpt_dir)
    if len(ckpt_lst)==0:
        epoch=0
        return net,optim,epoch
    ckpt_lst.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model=torch.load('./%s/%s'%(ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch=int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch


st_epoch=0#시작 에포크
net,optim,st_epoch=load(ckpt_dir=ckpt_dir,net=net,optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)
        input=input.to(device).float()

        output = net(input)

        # 손실계산
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("BATCH %04d / %04d | LOSS %.4f" %
              ( batch, num_batch_test, np.mean(loss_arr)))

        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id=num_batch_test*(batch-1)+j

            plt.imsave(os.path.join(result_dir, '%04d_label.png' % id), label[j].squeeze())
            plt.imsave(os.path.join(result_dir, '%04d_input.png' % id), input[j].squeeze())
            plt.imsave(os.path.join(result_dir, '%04d_output.png' % id), output[j].squeeze())


            # numpy로 저장
            ##np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            ##np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            ##np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      ( batch, num_batch_test, np.mean(loss_arr)))




