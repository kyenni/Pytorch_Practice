import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from data_loader import Dataset
from network import UNet
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


lr=1e-3
batch_size=4
num_epoch=55


data_train_dir='./drive/MyDrive/Colab Notebooks/archive (4)/train'
data_val_dir='./drive/MyDrive/Colab Notebooks/archive (4)/val'
ckpt_dir='./drive/MyDrive/Colab Notebooks/archive (4)/checkpoint'
log_dir='./drive/MyDrive/Colab Notebooks/archive (4)/log'
result_dir='./drive/MyDrive/Colab Notebooks/archive (4)/result'



device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')



if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)


transform_=transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.3),transforms.ToTensor(),transforms.Resize((512,512)),transforms.Normalize((0.5), (0.5))])
transform_t=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.3),
                                transforms.ToTensor(),
                                transforms.Resize((512,512))])
#transform_=transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512)),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#transform_t=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Resize((512,512))])
dataset_train=Dataset(data_train_dir,transform=transform_,target_transform=transform_t)
loader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=8)


dataset_val=Dataset(data_val_dir,transform=transform_,target_transform=transform_t)
loader_val=DataLoader(dataset_val,batch_size=batch_size,shuffle=False,num_workers=8)

net=UNet().to(device)

fn_loss=nn.BCEWithLogitsLoss().to(device)

optim=torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
num_data_train=len(dataset_train)
num_data_val=len(dataset_val)

num_batch_train=np.ceil(num_data_train/batch_size)
num_batch_val=np.ceil(num_data_val/batch_size)

fn_tonumpy=lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm=lambda x,mean,std : x*std+mean
fn_class=lambda x : 1.0*(x>0.5)

writer_train=SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val=SummaryWriter(log_dir=os.path.join(log_dir,'val'))


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
for epoch in range(st_epoch+1,num_epoch+1):
    net.train()
    loss_arr=[]
    ##forward pass
    for batch,data in enumerate(loader_train,1):
        label=data['label'].to(device)
        input=data['input'].to(device)
        input=input.to(device).float()


        output=net(input)


        #backward pass
        optim.zero_grad()

        loss=fn_loss(output,label)
        loss.backward()
  

        optim.step()

        #loss function computation
        loss_arr+=[loss.item()]
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f"%
              (epoch,num_epoch,batch,num_batch_train,np.mean(loss_arr)))

        
        ## 저장
        
        label=fn_tonumpy(label)
        input=fn_tonumpy(fn_denorm(input,0.5,0.5))
        output=fn_tonumpy(fn_class(output))
        
        

      
        for j in range(label.shape[0]):
           id=num_batch_train*(batch-1)+j

           plt.imsave(os.path.join(result_dir, '%04d_label.png' % id), label[j].squeeze())
           plt.imsave(os.path.join(result_dir, '%04d_input.png' % id), input[j].squeeze())
           plt.imsave(os.path.join(result_dir, '%04d_output.png' % id), output[j].squeeze())


        writer_train.add_image('label',label,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
    scheduler.step()    

    #로스를 텐서보드에 저장
    writer_train.add_scalar('loss',np.mean(loss_arr),epoch)


    #validation하는 부분 backward가 없다

    with torch.no_grad():
        net.eval()
        loss_arr=[]

        for batch,data in enumerate(loader_val,1):
            label=data['label'].to(device)
            input=data['input'].to(device)
            input=input.to(device).float()


            output=net(input)

            #손실계산
            loss=fn_loss(output,label) ##마이너스 값이 나오는 이유는 학습률이 너무커서
            loss_arr+=[loss.item()]

            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input,0.5,0.5))
            output = fn_tonumpy(fn_class(output))
         

            writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        #한번씩 epoch이 진행될 때마다 저장
        if epoch % 20==0:##5번마다 저장
            save(ckpt_dir=ckpt_dir,net=net,optim=optim,epoch=epoch)
    writer_train.close()
    writer_val.close()
