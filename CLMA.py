from models import *
import numpy as np
import torch
from attacks.CLL_mask import ComplementaryLabelAttack
from utils import new_data_utils as utils

np.seterr(divide='ignore', invalid='ignore')
device = torch.device('cuda:0')

torch.set_default_tensor_type(torch.FloatTensor)
model_name = "vgg_16"  # doesn't matter
batch_size = 20
image_size = 224  # 224 for vgg-16 and res-152, 299 for inc-v3, inc-res-v2
class_num = 1000
input_dir = "images/"
output_dir = "images_adv/"
rclm = "rclm/vgg16_rclm.pt"
epsilon = 16.  # 16. for vgg-16 and res-152, 16./255.*2. for inc-v3, inc-res-v2
step_size = 1.6  # 1.6 for vgg-16 and res-152, 1.6/255.*2. for inc-v3, inc-res-v2
steps = 10
ens = 30
prob = 0.7

# source model and RCLM have the same architecture
ori_model = Vgg_16()  # Vgg_16, Res_152, Inc_v3, IncRes_v2
adv_model = Vgg_16()  # Vgg_16, Res_152, Inc_v3, IncRes_v2

# load rclm parameter
ori_model.load_state_dict(torch.load(rclm))
ori_model.eval()
adv_model.eval()
ori_model.to(device)
adv_model.to(device)

internal = [i for i in range(29)]
# initialization
attack = ComplementaryLabelAttack(
    attack_model=adv_model, ori_model=ori_model, epsilon=epsilon, step_size=step_size, steps=steps, ens=ens, prob=prob)


def vgg_normalization(image):
    return image - [123.68, 116.78, 103.94]


def inv_vgg_normalization(image):
    return np.clip(image + [123.68, 116.78, 103.94], 0, 255)


def inception_normalization(image):
    return ((image / 255.) - 0.5) * 2


def inv_inception_normalization(image):
    return np.clip((image + 1.0) * 0.5 * 255, 0, 255)


count = 0
for images, names, labels in utils.load_image(input_dir, image_size, batch_size):
    count += batch_size
    if count % 100 == 0:
        print("Generating:", count)
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images)
    images = images.float()
    images = images.to(device)
    _, pred = ori_model.prediction(images, internal)
    cll_pred_labels = torch.argmax(pred, dim=1)
    cll_pred_labels = torch.eye(class_num)[cll_pred_labels, :]
    labels = labels - 1
    labels = torch.eye(class_num)[labels, :]
    images = images.to(device)
    cll_pred_labels = cll_pred_labels.to(device)
    labels = labels.to(device)

    adv = attack(images, cll_pred_labels,
                 index=count,
                 # vgg-16, conv4_3, attack_layer_idx=22
                 # res-152, block2, attack_layer_idx=1
                 # inc-v3, mixed_5b, attack_layer_idx=7
                 # inc-res-v2, conv_4a, attack_layer_idx=5
                 attack_layer_idx=22,
                 internal=internal
                 )
    adv_image = adv.cpu().numpy()
    labels = labels.cpu().numpy()
    adv_image = np.transpose(adv_image, (0, 2, 3, 1))
    # inv_vgg_normalization for for Vgg-16 and Res-152, inv_inception_normalization for inc-v3, inc-res-v2
    adv_image = inv_vgg_normalization(adv_image)
    adv_image = np.transpose(adv_image, (0, 3, 1, 2))
    utils.save_image(adv_image, names, output_dir)
