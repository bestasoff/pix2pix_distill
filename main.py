import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataset import AlignedDataset, normalize
from models import InceptionGenerator, Discriminator, Segmentator
from utils import ModuleProfiler, teacher_pruning
from train import train
from config import TrainingConfig, DataConfig

import albumentations

from albumentations.pytorch import ToTensorV2


def main():
    base_transform = albumentations.Compose([
        albumentations.Lambda(image=normalize, mask=normalize),
        ToTensorV2(transpose_mask=True)
    ])

    train_transform = albumentations.Compose([
        albumentations.Resize(276, 276),
        albumentations.RandomCrop(256, 256),
        albumentations.HorizontalFlip(),
        base_transform
    ])
    train_dataset = AlignedDataset(
        DataConfig.dataroot, transform=train_transform,
        direction=DataConfig.direction
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TrainingConfig.batch_size,
        num_workers=TrainingConfig.num_workers,
        shuffle=True
    )

    validation_transform = albumentations.Compose([
        base_transform
    ])
    validation_dataset = AlignedDataset(
        DataConfig.dataroot, is_train=False,
        transform=validation_transform,
        direction=DataConfig.direction
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=TrainingConfig.batch_size,
        num_workers=TrainingConfig.num_workers,
    )

    device = torch.device('cuda')

    teacher_generator = InceptionGenerator(
        3, 3, 64, 256, 6, [1, 3, 5],
        norm_layer=nn.BatchNorm2d
    ).eval().to(device)
    teacher_checkpoint = torch.load('best_net_G.pth', 'cpu')
    teacher_generator.load_state_dict(teacher_checkpoint)

    profiler = ModuleProfiler(teacher_generator, 256, 256)
    teacher_num_macs, teacher_num_params = profiler()

    cumputational_budget = teacher_num_macs // 10

    student_generator = teacher_pruning(teacher_generator, cumputational_budget)

    critic = Discriminator(6, use_affine=False).to(device)
    segmentator = Segmentator().to(device)

    student_generator, critic, generator_optimizer, critic_optimizer, history = \
        train(student_generator, teacher_generator, critic, segmentator, train_dataloader, validation_dataloader, device)

    torch.save({
        "student_generator": student_generator.state_dict(),
        "critic": critic.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "history": history
    }, "distill_checkpoint.ckpt")

if __name__ == "__main__":
    main()

