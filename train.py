import collections
import torch

from config import TrainingConfig
from torch import nn
from utils import get_feature_extractor
from tqdm import tqdm
from loss import KernelAlignmentLoss, HingeLoss


def train(student_generator, teacher_generator, critic, segmentator, train_dataloader, validation_dataloader, device):
    gan_criterion = HingeLoss().to(device)
    reconstruction_criterion = nn.L1Loss()
    distillation_criterion = KernelAlignmentLoss()

    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=TrainingConfig.lr,
        betas=(0.5, 0.999)
    )
    generator_optimizer = torch.optim.Adam(
        student_generator.parameters(), lr=TrainingConfig.lr,
        betas=(0.5, 0.999)
    )

    history = collections.defaultdict(list)

    for epoch in range(TrainingConfig.epoch):
        
        student_generator.train()
        for _, batch in enumerate(tqdm.tqdm(train_dataloader)):
            segmentation_mask, image = batch
            segmentation_mask = segmentation_mask.to(device)
            image = image.to(device)
            
            teacher_feature_extractor = get_feature_extractor(teacher_generator).to('cuda')
            student_feature_extractor = get_feature_extractor(student_generator).to('cuda')
            with torch.no_grad():
                teacher_features = teacher_feature_extractor(segmentation_mask)
                teacher_fake_image = teacher_features['up_sampling.8']
            student_features = student_feature_extractor(segmentation_mask)
            student_fake_image = student_features['up_sampling.8']
            
            
            ###################################
            ###### Update Student Critic ######
            ###################################
            
            for param in critic.parameters():
                param.requires_grad = True
                
            critic_response_on_real = critic(torch.cat([segmentation_mask, image], dim=1).detach())
            critic_respose_on_fake = critic(torch.cat([segmentation_mask, student_fake_image], dim=1).detach())
            critic_response_on_real_loss = gan_criterion(critic_response_on_real, True, True)
            critic_respose_on_fake_loss = gan_criterion(critic_respose_on_fake, False, True)
            total_critic_loss = 0.5 * (critic_response_on_real_loss + critic_respose_on_fake_loss)
            
            critic_optimizer.zero_grad()
            total_critic_loss.backward()
            critic_optimizer.step()
            for param in critic.parameters():
                param.requires_grad = False
                
            ###################################
            ##### Update Student Generator ####
            ###################################

            reconstruction_loss = reconstruction_criterion(teacher_fake_image, student_fake_image)
            
            critic_respose_on_fake = critic(torch.cat([segmentation_mask, student_fake_image], dim=1))
            generator_loss = gan_criterion(critic_respose_on_fake, True, False)
            distillation_loss = 0.0
            
            for layer in layers:
                distillation_loss += distillation_criterion(student_features[layer], teacher_features[layer])

            total_generator_loss = TrainingConfig.reconstruction_lambda * reconstruction_loss + \
                                TrainingConfig.generator_lambda * generator_loss + \
                                TrainingConfig.distillation_lambda * distillation_loss
            
            
            generator_optimizer.zero_grad()
            total_generator_loss.backward()
            generator_optimizer.step()
            
            ###################################
            ############# Logging #############
            ###################################
            losses = {
                'critic_real': critic_response_on_real_loss.item(),
                'critic_fake': critic_respose_on_fake_loss.item(),
                'reconstruction_loss': reconstruction_loss.item(),
                'generator_loss': generator_loss.item(),
                'distillation_loss': distillation_loss.item()
            }
            for k, v in losses.items():
                history[k].append(v)
                
        
        ###################################
        ############ Evaluation ###########
        ###################################
        num_classes = 19
        hist = np.zeros((num_classes, num_classes))
        student_generator.eval()
        for batch in tqdm.tqdm(validation_dataloader):
            segmentation_mask, image = batch
            segmentation_mask = segmentation_mask.to(device)
            image = image.to(device)
        
            with torch.inference_mode():
                fake_image = student_generator(segmentation_mask)
        
            # Denormalize
            image = image.mul(.5).add(0.5)
            fake_image = fake_image.mul(.5).add(0.5)
            
            predicted_segmentation_mask = segmentator(image).cpu()
            predicted_fake_segmentation_mask = segmentator(fake_image).cpu()
            
            hist += fast_hist(
                predicted_fake_segmentation_mask.numpy().flatten(),
                predicted_segmentation_mask.numpy().flatten(),
                num_classes
            )
        
        mean_iou = np.nanmean(per_class_iu(hist) * 100)
        median_iou = np.nanmedian(per_class_iu(hist) * 100)
        history['mean_iou'].append(mean_iou)
        history['median_iou'].append(median_iou)
    
    return student_generator, critic, generator_optimizer, critic_optimizer, history

