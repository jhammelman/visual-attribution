#!/bin/env python
import sys
import os
import utils
#import viz
import torch
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sequence_explainer import get_explainer
import argparse
from torch.nn import functional

alpha = ["A","C","G","T"]
def fasta2hot(fasta):
    flines = open(fasta).read().split('>')[1:]
    
    fasta_pos = [(f.strip().split()[0].split(':')[0],
                  int(f.strip().split()[0].split(':')[1].split('-')[0]),
                  int(f.strip().split()[0].split(':')[1].split('-')[1])) for f in flines]
    
    seqmat = np.zeros((len(flines),4,1000))
    for i,fline in enumerate(flines):
        seq = fline.strip().split('\n')[1].upper()
        for j,c in enumerate(seq):
            if c not in alpha:
                seqmat[i,:,j,:] = 0.25
            else:
                seqmat[i,alpha.index(c),j] = 1.0
    return fasta_pos,seqmat

parser = argparse.ArgumentParser()
parser.add_argument("model_weights")
parser.add_argument("fasta_sequences")
parser.add_argument("n_targets",type=int)
parser.add_argument("comparisons")
parser.add_argument("outpath")

opts=parser.parse_args()

from factorized import * 

                   
target_comparisons = []
for line in open(opts.comparisons):
    targets = line.strip().split('\t')
    target_comparisons.append(tuple((int(targets[0]),
                                    int(targets[1]))))
                              
methods = ['smooth_grad',
           'deeplift_rescale_zeros',
           'deeplift_rescale_neutral',
           'deeplift_rescale_shuffled',
           'saliency',
           'integrate_grad',
           'gradcam',
           'excitation_backprop',
           'contrastive_excitation_backprop']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(opts.outpath)
bed,X=fasta2hot(opts.fasta_sequences)
for target_set in target_comparisons:
    all_saliency_maps = {}
    for method in methods:
        print(target_set,method)
        if 'excitation' in method or 'gradcam' in method:
            values = np.zeros((X.shape[0],X.shape[2],1))
        else:
            values = np.zeros((X.shape[0],X.shape[2],X.shape[1]))
        for seq_index in range(X.shape[0]):
            # generate neural network architecture from factorized.py
            model = Net(1000,opts.n_targets)
                    
            #load model weights
            checkpoint = torch.load(opts.model_weights)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            #for keys in model.state_dict().keys():
            #    print(keys)    #2 does not show con2d list
            
            model.cuda()
            model.eval()
            #create explainer from sequence_explainer
            explainer = get_explainer(model,method)
            inp = torch.from_numpy(X[seq_index,:,:])
            inp = inp.float()
            inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True)
            target = torch.LongTensor([target_set[0]]).cuda()
            
            if 'difference' in method:
                target2 = torch.LongTensor([target_set[1]]).cuda()
                saliency = explainer.explain(inp, target, target2)
            else:
                saliency = explainer.explain(inp, target)
            print(saliency.cpu().numpy().shape)
            if 'excitation' in method or 'gradcam' in method:
                print(functional.interpolate(saliency.view(1,1,-1),size=X.shape[2],mode='linear').cpu().numpy().shape)
                values[seq_index,:,:] = np.transpose(functional.interpolate(saliency.view(1,1,-1),size=X.shape[2],mode='linear').cpu().numpy()[0,:,:])
            else:
                values[seq_index,:,:] = np.transpose(saliency.cpu().numpy()).reshape((-1,4))
          
        all_saliency_maps[method] = values
        
    with open(opts.outpath+'/'+str(target_set[0])+'-'+str(target_set[1])+'.pkl','wb') as f:
            pickle.dump(all_saliency_maps,f)

            
