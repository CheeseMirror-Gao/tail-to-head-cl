import torch
from utils.inc_net import IncrementalNet

import numpy as np
from torch.utils.data import Subset,DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

import torch.nn.functional as F

import argparse

model=IncrementalNet({"convnet_type":"resnet32"},pretrained=False)
model.update_fc(100)

with np.load("cifar100_tail_to_head_split.npz") as split:
    parser=argparse.ArgumentParser()
    parser.add_argument("--k",type=int,choices=[0,5],default=0)
    k=parser.parse_args().k
    stage2_indices=split["stage2_target_indices"].tolist()
    head_classes=split["head_classes"]
    target_classes=split["target_classes"]

saved_weights=torch.load(f"tail_to_head_k{k}_stage1_seed1993.pt",map_location="cpu")
model.load_state_dict(saved_weights)
print(f"已加载 k={k} 第一阶段模型")

dataset=CIFAR100("./data",train=True,download=False,transform=ToTensor())
stage2_dataset=Subset(dataset,stage2_indices)

stage2_generator=torch.Generator()
stage2_generator.manual_seed(1993)
stage2_loader=DataLoader(stage2_dataset,batch_size=128,shuffle=True,generator=stage2_generator)


print("第二阶段训练图片数",len(stage2_dataset))
print("第二阶段每轮批次数",len(stage2_loader))

device=torch.device("cuda:0")
model=model.to(device)

test_dataset=CIFAR100("./data",train=False,download=False,transform=ToTensor())
test_labels=np.array(test_dataset.targets)

def test_accuracy(class_ids):
    test_indices=np.flatnonzero(np.isin(test_labels,class_ids)).tolist()
    test_subset=Subset(test_dataset,test_indices)
    test_loader=DataLoader(test_subset,batch_size=128)

    model.eval()
    correct=0
    with torch.no_grad():
        for images,labels in test_loader:
            scores=model(images.to(device))["logits"]
            predictions=scores.argmax(dim=1)
            correct+=(predictions==labels.to(device)).sum().item()

    return 100*correct/len(test_subset)


print(f"第二阶段开训前，头部类准确率 {test_accuracy(head_classes):.2f}%")
print(f"第二阶段开训前，目标类准确率 {test_accuracy(target_classes):.2f}%")

model.train()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

for step,(images,labels) in enumerate(stage2_loader,start=1):
    images=images.to(device)
    labels=labels.to(device)

    scores=model(images)["logits"]
    loss=F.cross_entropy(scores,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%20==0 or step==len(stage2_loader):
        print(f"第二阶段已完成 {step}/{len(stage2_loader)}) 批",flush=True)

print(f"第二阶段训练后，头部类准确率 {test_accuracy(head_classes):.2f}%")
print(f"第二阶段训练后，目标类准确率 {test_accuracy(target_classes):.2f}%")

