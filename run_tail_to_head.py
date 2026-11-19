import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

import torch
import torch.nn.functional as F
from utils.inc_net import IncrementalNet

import argparse

with np.load("cifar100_tail_to_head_split.npz") as split:
    parser=argparse.ArgumentParser()
    parser.add_argument("--k",type=int,choices=[0,5],default=0)
    k=parser.parse_args().k
    stage1_indices=split[f"stage1_k{k}_indices"].tolist()
    head_classes=split["head_classes"]

dataset=CIFAR100("./data",train=True,download=False,transform=ToTensor())
stage1_k0_dataset=Subset(dataset,stage1_indices)
stage1_k0_loader=DataLoader(stage1_k0_dataset,batch_size=128,shuffle=True)

print("训练图片数",len(stage1_k0_dataset))
print("每个epoch批次数",len(stage1_k0_loader))

torch.manual_seed(1993)
model=IncrementalNet({"convnet_type":"resnet32"},pretrained=False)
model.update_fc(100)

device=torch.device("cuda:0")
model=model.to(device)
print("训练设备",device)

optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

for epoch in range(1,6):
    print(f"第 {epoch}/5 轮",flush=True)

    total_loss=0.0
    total_images=0
    num_batches=len(stage1_k0_loader)

    for step,(images,labels) in enumerate(stage1_k0_loader,start=1):
        images=images.to(device)
        labels=labels.to(device)

        scores=model(images)["logits"]
        loss=F.cross_entropy(scores,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size=len(labels)
        total_loss+=loss.item()*batch_size
        total_images+=batch_size

        if step%20==0 or step==num_batches:
            print(f"已完成 {step}/{num_batches}批",flush=True)

    print("本轮处理图片数",total_images)
    print("本轮平均训练损失",total_loss/total_images)


test_dataset=CIFAR100("./data",train=False,download=False,transform=ToTensor())
test_labels=np.array(test_dataset.targets)
head_test_indices=np.flatnonzero(np.isin(test_labels,head_classes)).tolist()

head_test_dataset=Subset(test_dataset,head_test_indices)
head_test_loader=DataLoader(head_test_dataset,batch_size=128,shuffle=False)

model.eval()
correct=0

with torch.no_grad():
    for images,labels in head_test_loader:
        scores=model(images.to(device))["logits"]
        predictions=scores.argmax(dim=1)
        correct+=(predictions==labels.to(device)).sum().item()

print("头部类测试图片数",len(head_test_dataset))
print(f"头部类测试准确率 {100*correct/len(head_test_dataset):.2f}%")

torch.save(model.state_dict(),f"tail_to_head_k{k}_stage1_seed1993.pt")
print(f"已保存 k={k} 第一阶段模型")
