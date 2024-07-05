import torch
import numpy as np
import torchvision
from torchvision import models, transforms
from stream_inspect import get_activations


def get_imnet_val_acts(extractor, module_name, valdir, selected_neuron=None, neuron_coord=None, subset=None, sort_acts=True, batch_size=1024, use_cuda=True):
    IMAGE_SIZE = 224
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    imagenet_data = torchvision.datasets.ImageFolder(valdir, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, drop_last=False)

    num_batches = len(dataloader)
    if subset is not None:
        num_batches = int(subset * len(dataloader))
    print(f"Running on {num_batches} batches.")
    input_acts = []
    activations = []
    most_images = []
    all_images = []
    act_list = []
    im_count = 0

    for j, (inputs, labels) in enumerate(dataloader):
        print(j)
        if j == num_batches:
            break

        if use_cuda and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        im_count += inputs.shape[0]

        norm_inputs = norm_transform(inputs)
        acts = get_activations(extractor, norm_inputs, module_name, neuron_coord, selected_neuron, use_center=True)
        activations.append(acts)

        inputs = inputs.cpu()
        for input in inputs:
            all_images.append(input)

    all_ord_list = np.arange(im_count).tolist()
    unrolled_act = [num for sublist in activations for num in sublist]
    if sort_acts:
        all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))
        return all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, input_acts
    else:
        return all_images, act_list, unrolled_act, None, None, input_acts