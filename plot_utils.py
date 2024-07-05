import torch
from scipy.stats import spearmanr, f_oneway
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_mix_histos(figs, fig_titles, plotdir):
    fig = make_subplots(rows=2, cols=4, subplot_titles=fig_titles, horizontal_spacing=0.05, vertical_spacing=0.1)

    for i in range(len(figs)+1):
        idx = i
        if i == 3 or i == len(figs):
            continue
        elif i > 3:
            idx -= 1

        fig.add_trace(figs[idx]["data"][0], row=(i//3)+1, col=(i%4)+1)

    fig.update_xaxes(title_text="Mix Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Mix Ratio", row=2, col=2)
    fig.update_xaxes(title_text="Mix Ratio", row=2, col=3)
    fig.update_xaxes(title_text="Mix Ratio", row=2, col=4)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_xaxes(tick0=0, dtick=1)
    fig.update_layout(margin={"b": 10, "l": 10, "r": 10, "t": 25})
    fig.update_annotations(font_size=12)
    fig.update_traces(marker=dict(color='#9467bd'))
    
    fig.write_image(f"{plotdir}/mix_hist_all.png", scale=2)


def plot_bn_acts_corrs(act_figs, mix_curve_figs, plotdir):
    fig = make_subplots(rows=2, cols=3, subplot_titles=['', '', '', '', '', ''], horizontal_spacing=0.05, vertical_spacing=0.1)
    count = 1
    for i in range(len(act_figs)):
        act_fig = act_figs[i]
        curve_fig = mix_curve_figs[i]
        title = act_fig.layout.title['text']
        if '1.1' in title or '2.1' in title or '3.1' in title:
            act_x = torch.tensor(act_fig['data'][0]['x'])
            act_y = torch.tensor(act_fig['data'][0]['y'])

            fig.add_trace(act_fig["data"][0], row=1, col=count)
            fig.update_traces(marker=dict(color='#9467bd', size=5), row=1, col=count)

            fig.add_hline(y=torch.mean(act_y[torch.nonzero(act_x >= 1)]), line_color="blue", name="Skip Mean", showlegend=True if count==1 else False, row=1, col=count)
            fig.add_hline(y=torch.mean(act_y[torch.nonzero(act_x < 1)]), line_color="red", name="Overwrite Mean", showlegend=True if count==1 else False, row=1, col=count)
            
            f_stat, p_val = f_oneway(act_y[torch.nonzero(act_x < 1)], act_y[torch.nonzero(act_x >= 1)])
            fig.add_annotation(text=f'F={f_stat[0]:.2f}<br>p={p_val[0]:.2e}', x=0.99, y=0.99, xref="x domain", yref="y domain", showarrow=False, row=1, col=count)
            
            curve_x = torch.tensor(curve_fig['data'][0]['x'])
            curve_y = torch.tensor(curve_fig['data'][0]['y'])

            fig.add_trace(curve_fig["data"][0], row=2, col=count)
            fig.update_traces(marker=dict(color='#9467bd', size=5), row=2, col=count)

            fig.add_hline(y=torch.mean(curve_y[torch.nonzero(curve_x >= 1)]), line_color="blue", row=2, col=count)
            fig.add_hline(y=torch.mean(curve_y[torch.nonzero(curve_x < 1)]), line_color="red", row=2, col=count)
            
            f_stat, p_val = f_oneway(curve_y[torch.nonzero(curve_x < 1)], curve_y[torch.nonzero(curve_x >= 1)])
            fig.add_annotation(text=f'F={f_stat[0]:.2f}<br>p={p_val[0]:.2e}', x=0.99, y=0.99, xref="x domain", yref="y domain", showarrow=False, row=2, col=count)

            fig.update_xaxes(title_text="Mix Ratio", row=2, col=count)
            fig.layout.annotations[count-1]['text'] = title

            count += 1

    fig.update_yaxes(title_text="Block Act to Input MEI", row=1, col=1)
    fig.update_yaxes(title_text="Pos Input-Block Tuning Curve 𝜌", row=2, col=1)
    fig.update_xaxes(tick0=0, dtick=1)
    fig.update_layout(margin={"b": 10, "l": 10, "r": 10, "t": 25})
    fig.update_annotations(font_size=12)
    fig.write_image(f"{plotdir}/bn_acts_corrs_all.png", scale=2)


def plot_weight_mags(bn_figs, conv_figs, plotdir):
    fig = make_subplots(rows=2, cols=3, subplot_titles=['', '', '', '', '', ''], horizontal_spacing=0.05, vertical_spacing=0.1)
    count = 1
    for i in range(len(bn_figs)):
        bn_fig = bn_figs[i]
        conv_fig = conv_figs[i]
        title = bn_fig.layout.title['text']
        if '1.1' in title or '2.1' in title or '3.1' in title:
            bn_x = torch.tensor(bn_fig['data'][0]['x'])
            bn_y = torch.tensor(bn_fig['data'][0]['y'])

            fig.add_trace(bn_fig["data"][0], row=1, col=count)
            fig.update_traces(marker=dict(color='#9467bd', size=5), row=1, col=count)
            r, pval = spearmanr(bn_x, bn_y)
            fig.add_annotation(text=f'𝜌={r:.2f}<br>p={pval:.2e}', x=0.99, y=0.99, xref="x domain", yref="y domain", showarrow=False, row=1, col=count)

            conv_x = torch.tensor(conv_fig['data'][0]['x'])
            conv_y = torch.tensor(conv_fig['data'][0]['y'])

            fig.add_trace(conv_fig["data"][0], row=2, col=count)
            fig.update_traces(marker=dict(color='#9467bd', size=5), row=2, col=count)
            r, pval = spearmanr(conv_x, conv_y)
            fig.add_annotation(text=f'𝜌={r:.2f}<br>p={pval:.2e}', x=0.99, y=0.99, xref="x domain", yref="y domain", showarrow=False, row=2, col=count)
            
            fig.update_xaxes(title_text="Mix Ratio", row=2, col=count)
            fig.layout.annotations[count-1]['text'] = title

            count += 1

    fig.update_yaxes(title_text="BN Weight Mag", row=1, col=1)
    fig.update_yaxes(title_text="Conv Mean Weight Mag", row=2, col=1)
    fig.update_xaxes(tick0=0, dtick=1)
    fig.update_layout(margin={"b": 10, "l": 10, "r": 10, "t": 25})
    fig.update_annotations(font_size=12)
    fig.write_image(f"{plotdir}/weight_mags_all.png", scale=2)


def plot_scale_percs(scale_percs, no_mix_percs, plotdir):
    x_cats = ["1.1", "2.0", "2.1", "3.0", "3.1", "4.0", "4.1"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_cats, y=scale_percs*100, name="All Criteria", showlegend=True, marker={"color": "purple"}))
    fig.add_trace(go.Bar(x=x_cats, y=no_mix_percs*100, name="No Mix Criterion", showlegend=True, marker={"color": "gold"}))
    fig.update_xaxes(title_text="Block", type='category')
    fig.update_yaxes(title_text=r"% of Channels Scale Invariant", tick0=0, dtick=5)
    fig.update_layout(margin={"b": 10, "l": 15, "r": 10, "t": 10})
    fig.write_image(f"{plotdir}/scale_invariance_percs.png", scale=2)