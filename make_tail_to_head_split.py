import numpy as np
from torchvision.datasets import CIFAR100

rng=np.random.default_rng(1993)
classes=rng.permutation(100)

head_classes=classes[:50]
target_classes=classes[50:]

print("头部类数量",len(head_classes))
print("目标类数量",len(target_classes))

dataset=CIFAR100("./data",train=True,download=False)
labels=np.array(dataset.targets)

stage1_target_indices=[]
stage2_target_indices=[]

for class_id in target_classes:
    class_indices=np.flatnonzero(labels==class_id)
    shuffled_indices=rng.permutation(class_indices)

    stage1_target_indices.extend(shuffled_indices[300:305])
    stage2_target_indices.extend(shuffled_indices[:300])

print("第一阶段目标类图片数量",len(stage1_target_indices))
print("第二阶段目标类图片数量",len(stage2_target_indices))

print("两阶段重复图片数量",len(np.intersect1d(stage1_target_indices,stage2_target_indices)))

stage1_head_indices=[]

for class_id in head_classes:
    class_indices=np.flatnonzero(labels==class_id)
    stage1_head_indices.extend(class_indices)

stage1_k0_indices=stage1_head_indices
stage1_k5_indices=stage1_head_indices+stage1_target_indices

print("k=0 第一阶段图片数量",len(stage1_k0_indices))
print("k=5 第一阶段图片数量",len(stage1_k5_indices))
print("k=5 两阶段重复图片数量",len(np.intersect1d(stage1_k5_indices,stage2_target_indices)))

np.savez(
    "cifar100_tail_to_head_split.npz",
    head_classes=head_classes,
    target_classes=target_classes,
    stage1_k0_indices=stage1_k0_indices,
    stage1_k5_indices=stage1_k5_indices,
    stage2_target_indices=stage2_target_indices,
)


