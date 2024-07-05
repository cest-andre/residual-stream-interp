import os
import argparse
import torch
from torchvision import transforms
from thingsvision import get_extractor
from PIL import Image
import numpy as np
from scipy.stats import spearmanr
import plotly.express as px
from plotly.subplots import make_subplots

from feature_grid import residual_grid, exemplar_grid
from plot_utils import plot_mix_histos, plot_bn_acts_corrs, plot_weight_mags, plot_scale_percs


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

block_list = [
    {
        'output_module': 'layer1.1',
        'input_module': 'layer1.0',
        'middle_module': 'layer1.1.bn2',
        'num_neurons': 64
    },
    {
        'output_module': 'layer2.0',
        'input_module': 'layer2.0.downsample.1',
        'middle_module': 'layer2.0.bn2',
        'num_neurons': 128
    },
    {
        'output_module': 'layer2.1',
        'input_module': 'layer2.0',
        'middle_module': 'layer2.1.bn2',
        'num_neurons': 128
    },
    {
        'output_module': 'layer3.0',
        'input_module': 'layer3.0.downsample.1',
        'middle_module': 'layer3.0.bn2',
        'num_neurons': 256
    },
    {
        'output_module': 'layer3.1',
        'input_module': 'layer3.0',
        'middle_module': 'layer3.1.bn2',
        'num_neurons': 256
    },
    {
        'output_module': 'layer4.0',
        'input_module': 'layer4.0.downsample.1',
        'middle_module': 'layer4.0.bn2',
        'num_neurons': 512
    },
    {
        'output_module': 'layer4.1',
        'input_module': 'layer4.0',
        'middle_module': 'layer4.1.bn2',
        'num_neurons': 512
    }
]


def get_activations(extractor, x, module_name, neuron_coord=None, channel_id=None, use_center=False):
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = torch.unsqueeze(x, 0)

    features = extractor.extract_features(
        batches=x,
        module_name=module_name,
        flatten_acts=False
    )

    if use_center:
        neuron_coord = features.shape[-1] // 2

    activations = features

    if neuron_coord is not None:
        activations = activations[:, :, neuron_coord, neuron_coord]

    if channel_id is not None:
        activations = activations[:, channel_id]

    return activations


def measure_invariance(extractor, fzdir, curvedir, model_name, output_module, input_module, middle_module, neuron, variant="scale", imnet_val=False, show_results=False):
    top_img = None
    if imnet_val:
        img_path = os.path.join(curvedir, model_name, input_module, f"{input_module}_neuron{neuron}")
        top_img = [norm_trans(Image.open(os.path.join(img_path, file))) for file in os.listdir(img_path) if ".png" in file]
        top_img = torch.stack(top_img)
    else:
        top_img = Image.open(os.path.join(fzdir, model_name, input_module, f"unit{neuron}", "0_distill_center.png"))
        top_img = norm_trans(top_img)

    variant_trans = None
    if variant == "scale":
        variant_trans = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.Resize(224)
        ])
    elif variant == "flip":
        variant_trans = transforms.RandomHorizontalFlip(p=1)

    mid_base_act = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)
    mid_variant_act = get_activations(extractor, variant_trans(top_img), middle_module, channel_id=neuron, use_center=True)

    mid_act_delta = np.mean(np.clip(mid_variant_act, 0, None)) - np.mean(np.clip(mid_base_act, 0, None))

    if imnet_val:
        img_path = os.path.join(curvedir, model_name, middle_module, f"{middle_module}_neuron{neuron}")
        top_img = [norm_trans(Image.open(os.path.join(img_path, file))) for file in os.listdir(img_path) if ".png" in file]
        top_img = torch.stack(top_img)
    else:
        top_img = Image.open(os.path.join(fzdir, model_name, middle_module, f"unit{neuron}", "0_distill_center.png"))
        top_img = norm_trans(top_img)

    if variant == "scale":
        variant_trans = transforms.Compose([
            transforms.Resize(112),
            transforms.Pad(56, padding_mode='reflect')
        ])

    in_base_act = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    in_variant_act = get_activations(extractor, variant_trans(top_img), input_module, channel_id=neuron, use_center=True)

    in_act_delta = np.mean(np.clip(in_variant_act, 0, None)) - np.mean(np.clip(in_base_act, 0, None))

    if show_results:
        print(f"\nMid base Act:  {mid_base_act[0]},  Mid {variant} act:  {mid_variant_act[0]}")
        print(f"\nIn base Act:  {in_base_act[0]},  In {variant} act:  {in_variant_act[0]}")

    return mid_act_delta, in_act_delta



def stream_inspect(extractor, fzdir, curvedir, model_name, output_module, input_module, middle_module, neuron, imnet_val=False, show_results=False):
    top_img = None
    if imnet_val:
        img_path = os.path.join(curvedir, model_name, output_module, f"{output_module}_neuron{neuron}")
        top_img = [norm_trans(Image.open(os.path.join(img_path, file))) for file in os.listdir(img_path) if ".png" in file]
        top_img = torch.stack(top_img)
    else:
        top_img = Image.open(os.path.join(fzdir, model_name, output_module, f"unit{neuron}", "0_distill_center.png"))
        top_img = norm_trans(top_img)

    out_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    out_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    out_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if imnet_val:
        img_path = os.path.join(curvedir, model_name, input_module, f"{input_module}_neuron{neuron}")
        top_img = [norm_trans(Image.open(os.path.join(img_path, file))) for file in os.listdir(img_path) if ".png" in file]
        top_img = torch.stack(top_img)
    else:
        top_img = Image.open(os.path.join(fzdir, model_name, input_module, f"unit{neuron}", "0_distill_center.png"))
        top_img = norm_trans(top_img)

    in_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    in_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    in_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if imnet_val:
        img_path = os.path.join(curvedir, model_name, middle_module, f"{middle_module}_neuron{neuron}")
        top_img = [norm_trans(Image.open(os.path.join(img_path, file))) for file in os.listdir(img_path) if ".png" in file]
        top_img = torch.stack(top_img)
    else:
        top_img = Image.open(os.path.join(fzdir, model_name, middle_module, f"unit{neuron}", "0_distill_center.png"))
        top_img = norm_trans(top_img)

    mid_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    mid_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    mid_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if show_results:
        print("---OUTPUT MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(out_out_acts)}, Pos act: {np.mean(np.clip(out_out_acts, 0, None))}, Neg act: {np.mean(np.clip(out_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(out_in_acts)}, Pos act: {np.mean(np.clip(out_in_acts, 0, None))}, Neg act: {np.mean(np.clip(out_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(out_mid_acts)}, Pos act: {np.mean(np.clip(out_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(out_mid_acts, None, 0))}")
        print("-------------------------------")

        print("---INPUT MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(in_out_acts)}, Pos act: {np.mean(np.clip(in_out_acts, 0, None))}, Neg act: {np.mean(np.clip(in_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(in_in_acts)}, Pos act: {np.mean(np.clip(in_in_acts, 0, None))}, Neg act: {np.mean(np.clip(in_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(in_mid_acts)}, Pos act: {np.mean(np.clip(in_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(in_mid_acts, None, 0))}")
        print("-------------------------------")

        print("---MIDDLE MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(mid_out_acts)}, Pos act: {np.mean(np.clip(mid_out_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(mid_in_acts)}, Pos act: {np.mean(np.clip(mid_in_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(mid_mid_acts)}, Pos act: {np.mean(np.clip(mid_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_mid_acts, None, 0))}")
        print("-------------------------------")

    return {
               "out_out_acts": np.mean(out_out_acts), "out_in_acts": np.mean(out_in_acts), "out_mid_acts": np.mean(out_mid_acts),
               "in_out_acts": np.mean(in_out_acts), "in_in_acts": np.mean(in_in_acts), "in_mid_acts": np.mean(in_mid_acts),
               "mid_out_acts": np.mean(mid_out_acts), "mid_in_acts": np.mean(mid_in_acts), "mid_mid_acts": np.mean(mid_mid_acts)
           }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--fzdir', type=str)
    parser.add_argument('--curvedir', type=str)
    parser.add_argument('--plotdir', type=str)
    parser.add_argument('--run_mode', type=str)
    args = parser.parse_args()

    network = args.network
    fzdir = args.fzdir
    curvedir = args.curvedir
    plotdir = args.plotdir
    run_mode = args.run_mode

    extractor = get_extractor(
        model_name=network,
        source='torchvision',
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=True
    )

    if run_mode == "spectrum":
        top_mixes = []
        mix_hist_figs = []
        bn_act_figs = []
        bn_mag_figs = []
        conv_mag_figs = []
        mix_curve_figs = []
        fig_titles = []
        for block in block_list:
            mixes = []
            in_mid_acts = []
            bn_weights = []
            conv_weights = []
            curve_corrs = []
            imnet_val = False
            show_results = False
            for n in range(block['num_neurons']):
                acts = stream_inspect(extractor, fzdir, curvedir, network, block['output_module'], block['input_module'], block['middle_module'], n, imnet_val=imnet_val, show_results=show_results)

                assert acts['mid_out_acts'] >= 0 or acts['in_out_acts'] >= 0, f"Neither mix component is above 0.  Rerun FZ on neuron {n}."

                converted_mix = None
                if acts['mid_out_acts'] <= 0:
                    converted_mix = 10
                elif acts['in_out_acts'] <= 0:
                    converted_mix = 0
                else:
                    converted_mix = acts['in_out_acts'] / acts['mid_out_acts']

                mixes.append(converted_mix)
                in_mid_acts.append(acts['in_mid_acts'])

                bn_weight = extractor.model.state_dict()[block['middle_module'] + '.weight'][n]
                bn_weights.append(torch.abs(bn_weight).cpu())

                conv_weight = torch.flatten(extractor.model.state_dict()[block['middle_module'].replace('bn', 'conv') + '.weight'][n])
                conv_weights.append(torch.mean(torch.abs(conv_weight)).cpu())

                if '1.1' in block['output_module'] or '2.1' in block['output_module'] or '3.1' in block['output_module']:
                    input_name = block['input_module'].replace('.prerelu_out', '')
                    bn_name = block['middle_module']
                    input_curve = np.load(os.path.join(curvedir, network, input_name, f"{input_name}_unit{n}_unrolled_act.npy"))
                    bn_curve = np.load(os.path.join(curvedir, network, bn_name, f"{bn_name}_unit{n}_unrolled_act.npy"))

                    curve_corrs.append(spearmanr(input_curve[np.nonzero(input_curve > 0)], bn_curve[np.nonzero(input_curve > 0)])[0])

            in_mid_acts = torch.tensor(in_mid_acts)
            bn_weights = torch.tensor(bn_weights)
            conv_weights = torch.tensor(conv_weights)
            mixes = torch.clamp(torch.tensor(mixes), max=5)

            top_mixes.append({f"top mixes {block['output_module']}": torch.topk(torch.abs(mixes - 1), 3, largest=False)})

            fig_title = f"Output: {block['output_module'].replace('.prerelu_out', '').replace('layer', '')}"
            fig_titles.append(fig_title)

            counts, bins = np.histogram(mixes.numpy(), bins=np.arange(0, 5.25, 0.25))
            bins = 0.5 * (bins[:-1] + bins[1:])
            fig = px.bar(x=bins, y=counts)
            mix_hist_figs.append(fig)

            fig = px.scatter(x=mixes, y=in_mid_acts, title=f"Output: {block['output_module'].replace('.prerelu_out', '').replace('layer', '')}")
            bn_act_figs.append(fig)

            fig = px.scatter(x=mixes, y=bn_weights, title=f"Output: {block['output_module'].replace('.prerelu_out', '').replace('layer', '')}")
            bn_mag_figs.append(fig)

            fig = px.scatter(x=mixes, y=conv_weights, title=f"Output: {block['output_module'].replace('.prerelu_out', '').replace('layer', '')}")
            conv_mag_figs.append(fig)

            if '1.1' in block['output_module'] or '2.1' in block['output_module'] or '3.1' in block['output_module']:
                curve_corrs = torch.tensor(curve_corrs)
                fig = px.scatter(x=mixes, y=curve_corrs, title=f"Output: {block['output_module'].replace('.prerelu_out', '').replace('layer', '')}")
                mix_curve_figs.append(fig)
            else:
                mix_curve_figs.append(None)

        print(top_mixes)

        #   Leave empty subplot title blank.
        fig_titles.insert(3, '')
        plot_mix_histos(mix_hist_figs, fig_titles, plotdir)

        plot_bn_acts_corrs(bn_act_figs, mix_curve_figs, plotdir)
        plot_weight_mags(bn_mag_figs, conv_mag_figs, plotdir)
    elif run_mode == 'invariance':
        scale_percs = []
        no_mix_percs = []
        scale_mixes = []
        in_deltas = []
        mid_deltas = []
        chans = []
        for block in block_list:
            scale_mid_measures = []
            scale_in_measures = []           
            mixes = []
            imnet_val = False
            show_results = False
            for n in range(block['num_neurons']):
                acts = stream_inspect(extractor, fzdir, curvedir, network, block['output_module'], block['input_module'], block['middle_module'], n, imnet_val=imnet_val, show_results=show_results)

                assert acts['mid_out_acts'] >= 0 or acts['in_out_acts'] >= 0, f"Neither mix component is above 0.  Rerun FZ on neuron {n}."

                mid_act_delta, in_act_delta = measure_invariance(extractor, fzdir, curvedir, network, block['output_module'], block['input_module'], block['middle_module'], n, imnet_val=imnet_val, show_results=show_results)
                #   NOTE:  store deltas as a perc of max activation to control for different act ranges in the two modules.
                scale_mid_measures.append(mid_act_delta / acts['mid_mid_acts'])
                scale_in_measures.append(in_act_delta / acts['in_in_acts'])

                converted_mix = None
                if acts['mid_out_acts'] <= 0:
                    converted_mix = 10
                elif acts['in_out_acts'] <= 0:
                    converted_mix = 0
                else:
                    converted_mix = acts['in_out_acts'] / acts['mid_out_acts']

                mixes.append(converted_mix)

            mixes = torch.tensor(mixes)

            scale_mid_measures = torch.tensor(scale_mid_measures)
            scale_in_measures = torch.tensor(scale_in_measures)

            factor = 100
            scale_mid_copy = torch.clone(scale_mid_measures)
            scale_in_copy = torch.clone(scale_in_measures)
            scale_mid_copy[torch.nonzero(torch.logical_or(scale_mid_copy <= 0, torch.logical_or(mixes <= (1/factor), mixes >= factor)))] = -1e5
            scale_in_copy[torch.nonzero(torch.logical_or(scale_in_copy <= 0, torch.logical_or(mixes <= (1/factor), mixes >= factor)))] = -1e5
            scale_top_cross_delta = scale_mid_copy + scale_in_copy
            top_scales, top_scale_channels = torch.topk(scale_top_cross_delta, torch.nonzero((scale_mid_copy + scale_in_copy) > 0).shape[0])
            scale_percs.append(top_scales.shape[0] / block['num_neurons'])

            #   No mix criterion.
            scale_mid_copy = torch.clone(scale_mid_measures)
            scale_in_copy = torch.clone(scale_in_measures)
            scale_mid_copy[torch.nonzero(scale_mid_copy <= 0)] = -1e5
            scale_in_copy[torch.nonzero(scale_in_copy <= 0)] = -1e5
            scale_top_cross_delta = scale_mid_copy + scale_in_copy
            no_mix_top_scales, no_mix_top_scale_channels = torch.topk(scale_top_cross_delta, torch.nonzero((scale_mid_copy + scale_in_copy) > 0).shape[0])
            no_mix_percs.append(no_mix_top_scales.shape[0] / block['num_neurons'])

            #   Extract top 3 channels for visualization.
            if top_scales.shape[0] > 3:
                top_scale_channels = top_scale_channels[torch.topk(top_scales, 3)[1]]

            scale_mixes.append({f"scale mix {block['output_module']}": mixes[top_scale_channels]})
            in_deltas.append({f"in deltas {block['output_module']}": scale_in_measures[top_scale_channels]})
            mid_deltas.append({f"mid deltas {block['output_module']}": scale_mid_measures[top_scale_channels]})
            chans.append({f"all channels {block['output_module']}": top_scale_channels})

            # continue

            for c in top_scale_channels:
                exemplar_grid(fzdir, network, block['input_module'], block['middle_module'], block['output_module'], c, fzdir, val_dir=curvedir)
                # scale_grid = residual_grid(fzdir, network, block['input_module'], block['middle_module'], block['output_module'], top_scale_channels)
                # scale_grid.show()

        print(chans)
        print(scale_mixes)
        print(in_deltas)
        print(mid_deltas)
        plot_scale_percs(torch.tensor(scale_percs), torch.tensor(no_mix_percs), plotdir)