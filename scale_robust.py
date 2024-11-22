import argparse
from random import choice
import torch
from torchvision import models
import numpy as np

from imnet_val import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valdir', type=str)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    #   Top 9 imnet val intersected with center FZ.
    scale_inv_2_1 = [108, 36, 13, 21, 60, 52, 93, 5, 58]
    scale_inv_3_1 = [113, 178, 22, 215, 204, 25, 29, 154, 200]

    #   Top 9 imnet val
#     scale_inv_3_1 = [122, 198, 88, 114, 69, 155, 87, 204, 179, 250, 66, 120, 225, 175,\
# 218, 135, 189, 67, 188, 248, 113, 22, 29, 215, 216, 207, 178, 200, 14, 77, 31, 137, 107,\
# 169, 209, 6, 20, 100, 92, 237, 184, 18, 145, 253, 26, 249, 131, 206, 99, 224, 251, 245,\
# 254, 147, 191, 119, 180, 129, 133, 164, 57, 25, 98, 138, 141, 150, 154]

    mean_inv_acts = []
    for c in scale_inv_3_1:
        curve = torch.tensor(np.load(f'/media/andrelongon/DATA/tuning_curves/resnet18/layer3.1.prerelu_out/layer3.1.prerelu_out_unit{c}_all_act_list.npy'))
        curve_mean = torch.mean(torch.tensor(curve))
        curve_mean = torch.clamp(curve_mean, min=0)
        mean_inv_acts.append(curve_mean)

    mean_inv_acts = torch.tensor(mean_inv_acts, dtype=torch.float32).cuda()
    all_mean_inv_acts = [
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': mean_inv_acts}],
        [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
    ]

    scale_lesions = [
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': scale_inv_3_1}],
        [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
    ]

    # model = models.resnet18(False, lesion=False).cuda()
    # model.load_state_dict(torch.load('/media/andrelongon/DATA/imnet_weights/vision_resnet18/intact/model_89.pth')['model'])

    # model = models.resnet18(False, lesion=True).cuda()
    # model.load_state_dict(torch.load('/media/andrelongon/DATA/imnet_weights/vision_resnet18/lesion/model_89.pth')['model'])

    #   2.1 no-clamp mean ablate accs (clamped to zero AFTER mean is taken)
    # vanilla_acc = 67.576
    # up_accs = [65.808, 64.286, 61.716, 57.74, 51.286]

    #   3.1 no-clamp mean ablate accs (clamped to zero AFTER mean is taken)
    vanilla_acc = 68.63
    up_accs = [67.374, 65.74, 63.178, 59.272, 52.934]

    model = models.resnet18(True, lesion=True).cuda()
    model.eval()

    trials = 0
    no_scale_results = []
    while trials < 10:
        print(f"\n\nTRIAL {trials+1}")

        # rand_2_1 = []
        # while len(rand_2_1) < len(scale_inv_2_1):
            # rand_2_1.append(choice([i for i in range(128) if i not in scale_inv_2_1 and i not in rand_2_1]))

        rand_3_1 = []
        while len(rand_3_1) < len(scale_inv_3_1):
            rand_3_1.append(choice([i for i in range(256) if i not in scale_inv_3_1 and i not in rand_3_1]))

        mean_rand_acts = []
        for c in rand_3_1:
            curve = np.load(f'/media/andrelongon/DATA/tuning_curves/resnet18/layer3.1.prerelu_out/layer3.1.prerelu_out_unit{c}_all_act_list.npy')
            curve_mean = torch.mean(torch.tensor(curve))
            curve_mean = torch.clamp(curve_mean, min=0)
            mean_rand_acts.append(curve_mean)

        mean_rand_acts = torch.tensor(mean_rand_acts, dtype=torch.float32).cuda()
        all_mean_rand_acts = [
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': mean_rand_acts}],
            [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
        ]

        rand_lesions = [
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': rand_3_1}],
            [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
        ]

        # print('No scale')
        # scale_lesion_acc = validate(model, args.valdir, lesions=scale_lesions, mean_ablates=all_mean_inv_acts)
        
        rand_lesion_acc = validate(model, args.valdir, lesions=rand_lesions, mean_ablates=all_mean_inv_acts)
        print(f"No Scale lesion delta: {vanilla_acc - rand_lesion_acc}\n")

        #   We only want to test scale robustness on random channels which are MORE important for general object recognition.
        if vanilla_acc - rand_lesion_acc < 0:
            print('Rand lesion is less damaging without scale.  Skipping')
            continue

        no_scale_results.append(rand_lesion_acc)
        
        up_results = []
        down_black_results = []
        down_gray_results = []
        for i in range(1, 6, 1):
            frac = i / 10
            # print(f"Scale-Up {2-frac}")
            # scale_acc = validate(model, args.valdir, scale=2-frac, lesions=scale_lesions, mean_ablates=all_mean_inv_acts)

            rand_lesion_acc = validate(model, args.valdir, scale=2-frac, lesions=rand_lesions, mean_ablates=all_mean_inv_acts)
            print(f"Scale-Up {2-frac} lesion delta: {up_accs[i-1] - rand_lesion_acc}\n")
            up_results.append(rand_lesion_acc)
            
            # print(f"Scale-Down Black {1-frac}")
            # scale_acc = validate(model, args.valdir, scale=1-frac, lesions=scale_lesions, mean_ablates=None, fill=0)

            # rand_lesion_acc = validate(model, args.valdir, scale=1-frac, lesions=rand_lesions, mean_ablates=all_mean_rand_acts, fill=0)
            # print(f"Scale-Down {1-frac} Black lesion delta: {down_black_accs[i-1] - rand_lesion_acc}\n")
            # down_black_results.append(rand_lesion_acc)

            # print(f"Scale-Down Gray {1-frac}")
            # scale_acc = validate(model, args.valdir, scale=1-frac, lesions=scale_lesions, mean_ablates=None, fill=127)

            # rand_lesion_acc = validate(model, args.valdir, scale=1-frac, lesions=rand_lesions, mean_ablates=all_mean_rand_acts, fill=127)
            # print(f"Scale-Down {1-frac} Gray lesion delta: {down_gray_accs[i-1] - rand_lesion_acc}\n")
            # down_gray_results.append(rand_lesion_acc)

        np.save(f'/media/andrelongon/DATA/scale_robust_results/layer3.1/up_rand_mean_trial_{trials}.npy', np.array(up_results))
        # np.save(f'/media/andrelongon/DATA/scale_robust_results/layer3.1/down_black_rand_zero_trial_{trials}.npy', np.array(down_black_results))
        # np.save(f'/media/andrelongon/DATA/scale_robust_results/layer3.1/down_gray_rand_zero_trial_{trials}.npy', np.array(down_gray_results))
        trials += 1

    np.save(f'/media/andrelongon/DATA/scale_robust_results/layer3.1/no_scale_rand_mean_all_trials.npy', np.array(no_scale_results))