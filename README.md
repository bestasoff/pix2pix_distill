# pix2pix_distill
Pix2Pix model compression and distillation

Implementation of paper **"Teachers Do More Than Teach: Compressing Image-to-Image Models"** (https://arxiv.org/abs/2103.03467).

Teacher generator is being pruned using binary search over BatchNorm scaling factors to meet certain computational budget requirements.
