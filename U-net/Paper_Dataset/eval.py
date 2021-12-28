import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

lr=1e-3
batch_size=4
num_epoch=100
##colab할때는 colab기준으로 파일 이름 변경 필요
data_dir='./drive/MyDrive/Colab Notebooks/unet/tif_data'
ckpt_dir='./drive/MyDrive/Colab Notebooks/unet/checkpoint'
log_dir='./drive/MyDrive/Colab Notebooks/unet/log'
result_dir='./drive/MyDrive/Colab Notebooks/unet/results'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir,'png'))
    os.makedirs(os.path.join(result_dir,'numpy'))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

##네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        ## convolutional, batch, relu 함수 만들기
        def CBR2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                               stride=stride,padding=padding,bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbr=nn.Sequential(*layers)

            return cbr

        ##필요한 레이어 선언 contracting path
        self.enc1_1=CBR2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        ##Expansive path

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4=nn.ConvTranspose2d(in_channels=512,out_channels=512,
                                        kernel_size=2,stride=2,padding=0,bias=True)

        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)##여기 안줄어드는데..
        self.unpool3=nn.ConvTranspose2d(in_channels=256,out_channels=256,
                                        kernel_size=2,stride=2,padding=0,bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self,x):
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1=self.dec5_1(enc5_1)
        unpool4=self.unpool4(dec5_1)

        cat4=torch.cat((unpool4,enc4_2),dim=1)##0은 배치, 1=channel,2=height,3=width
        dec4_2=self.dec4_2(cat4)
        dec4_1=self.dec4_1(dec4_2)
        unpool3=self.unpool3(dec4_1)

        cat3=torch.cat((unpool3,enc3_2),dim=1)
        dec3_2=self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)

        cat2=torch.cat((unpool2,enc2_2),dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)

        cat1=torch.cat((unpool1,enc1_2),dim=1)
        dec1_2=self.dec1_2(cat1)
        dec1_1=self.dec1_1(dec1_2)

        x=self.fc(dec1_1)

        return x

##Data loader 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.transform=transform

        lst_data=os.listdir(self.data_dir)

        lst_label=[f for f in lst_data if f.startswith('label')]
        lst_input=[f for f in lst_data if f.startswith('input')]
        print(lst_label)

        lst_label.sort()
        lst_input.sort()

        self.lst_label=lst_label #함수의 변수로 만들기
        self.lst_input=lst_input

    def __len__(self):
        return len(self.lst_label)

    ##index에 해당하는 파일을 로드해서 return하는 함수
    def __getitem__(self,index):
        label=np.load(os.path.join(self.data_dir,self.lst_label[index]))
        input=np.load(os.path.join(self.data_dir,self.lst_input[index]))

        label=label/255.0
        input=input/255.0

    ##네트워크에 들어가는 데이터는 3차원이어야 한다.
        if label.ndim==2:
            label=label[:,:,np.newaxis]
        if input.ndim==2:
            input=input[:,:,np.newaxis]

        data={'input':input,'label':label}

        if self.transform:
            data=self.transform(data)
            ##트랜스폼 함수가 정의되어있다면 통과해야한다.

        return data



##transform 함수

##numpy에서 tensor로 변경
class ToTensor(object):
    def __call__(self,data):
        label,input=data['label'],data['input']
        ##image의 넘파이 차원(y,x,ch)
        ##torch에서는 tensor차원(ch,y,x)로 변경 필요
        label=label.transpose((2,0,1)).astype(np.float32)
        input=input.transpose((2,0,1)).astype(np.float32)
        ##넘파이를 텐서로 넘기기from_numpy
        data={'label':torch.from_numpy(label),'input':torch.from_numpy(input)}

        return data
class Normalization(object):
    def __init__(self,mean=0.5,std=0.5):
        self.mean=mean
        self.std=std
    def __call__(self,data):
        label,input=data['label'],data['input']

        input=(input-self.mean)/self.std
        ##라벨은 명도가 1인 흑백 이미지이므로 정규화 하면 안된다


        data={'label':label,'input':input}

        return data

class RandomFlip(object):
    def __call__(self,data):
        label, input = data['label'], data['input']

        if np.random.rand()>0.5:
            label=np.fliplr(label)
            input=np.fliplr(input)

        if np.random.rand()>0.5:
            label=np.flipud(label)
            input=np.flipud(input)

        data={'label':label,'input':input}

        return data



##네트워크 학습하기
#트레이닝 데이터를 불러올 때 이러한 트랜스폼 함수를 불러온다
transform=transforms.Compose([Normalization(mean=0.5,std=0.5),ToTensor()])
##데이터셋 불러오기
dataset_test=Dataset(data_dir=os.path.join(data_dir,'test'),transform=transform)
loader_test=DataLoader(dataset_test,batch_size=batch_size,shuffle=False,num_workers=8)

##네트워크 생성
net=UNet().to(device)

##손실함수 정의
fn_loss=nn.BCEWithLogitsLoss().to(device)
##옵티마이저 설정
optim=torch.optim.Adam(net.parameters(),lr=lr)

#그박에 부수적인
num_data_test=len(dataset_test)

num_batch_test=np.ceil(num_data_test/batch_size)

##output저장함수

fn_tonumpy=lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)##from tensor to numpy
fn_denorm=lambda x,mean,std : (x*std)+mean ##정규화 다시inverse
fn_class=lambda x:1.0*(x>0.5) ##바이너리 분류

#네트워크 저장하기
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
                "./%s/model_epoch%d.pth"%(ckpt_dir,epoch))

##네트워크 불러오기

def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir):
        epoch=0
        return net,optim,epoch

    ckpt_lst=os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model=torch.load('./%s/%s'%(ckpt_dir,ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch=int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net,optim,epoch

##트레이닝 수행

st_epoch=0#시작 에포크
net,optim,st_epoch=load(ckpt_dir=ckpt_dir,net=net,optim=optim)
    ##test하는 부분 backward가 없다

with torch.no_grad():
    net.eval()
    loss_arr=[]

    for batch,data in enumerate(loader_test,1):
        label=data['label'].to(device)
        input=data['input'].to(device)

        output=net(input)

        #손실계산
        loss=fn_loss(output,label)
        loss_arr+=[loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id=num_batch_test*(batch-1)+j

            #png파일로 저장하는 방법
            plt.imsave(os.path.join(result_dir,'png','label_%04d.png'%id),label[j].squeeze(),cmap='gray')
            plt.imsave(os.path.join(result_dir,'png','input_%04d.png'%id),input[j].squeeze(),cmap='gray')
            plt.imsave(os.path.join(result_dir,'png','output_%04d.png'%id),output[j].squeeze(),cmap='gray')

            #numpy로 저장
            np.save(os.path.join(result_dir,'numpy','label_%04d.npy'%id),label[j].squeeze())
            np.save(os.path.join(result_dir,'numpy','input_%04d.npy'%id),input[j].squeeze())
            np.save(os.path.join(result_dir,'numpy','output_%04d.npy'%id),output[j].squeeze())

##평균 손실함수 값 프린트
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      ( batch, num_batch_test, np.mean(loss_arr)))

