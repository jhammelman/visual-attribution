from explainer import backprop as bp
from explainer import deeplift as df
from explainer import gradcam as gc
from explainer import patterns as pt
from explainer import ebp
from explainer import real_time as rt


def get_explainer(model,name):
    methods = {
        'vanilla_grad': bp.VanillaGradExplainer,
        'grad_x_input': bp.GradxInputExplainer,
        'saliency': bp.SaliencyExplainer,
        'integrate_grad': bp.IntegrateGradExplainer,
        'deconv': bp.DeconvExplainer,
        'guided_backprop': bp.GuidedBackpropExplainer,
        'deeplift_rescale': df.DeepLIFTRescaleExplainer,
        'gradcam': gc.GradCAMExplainer,
        'pattern_net': pt.PatternNetExplainer,
        'pattern_lrp': pt.PatternLRPExplainer,
        'excitation_backprop': ebp.ExcitationBackpropExplainer,
        'contrastive_excitation_backprop': ebp.ContrastiveExcitationBackpropExplainer,
        'vanilla_difference':bp.VanillaDifferenceGradExplainer
    }
    
    if name == 'smooth_grad':
        base_explainer = methods['vanilla_grad'](model)
        explainer = bp.SmoothGradExplainer(base_explainer)
        
    elif name == 'smooth_difference':
        base_explainer = methods['vanilla_difference'](model)
        explainer = bp.SmoothGradExplainer(base_explainer)

    elif name == 'excitation_backprop' or name == 'gradcam':
        explainer = methods[name](model,['layer1'])

    elif name == 'contrastive_excitation_backprop':
        explainer = methods[name](model,
                                  intermediate_layer_keys=['layer1'], 
                                  output_layer_keys=['layer1'],  
                                  final_linear_keys=['layer2'])
    
    else:
        explainer = methods[name](model)

    return explainer


def get_heatmap(saliency):
    saliency = saliency.squeeze()

    if len(saliency.size()) == 2:
        return saliency.abs().cpu().numpy()
    else:
        return saliency.abs().max(0)[0].cpu().numpy()
