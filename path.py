import numpy as np
from sklearn.neighbors import BallTree

def dijkstra(u,v,distances,indices):
    sptset = np.zeros((distances.shape[0]))

    sptset[u] = 1.0
    prev_i = [u]
    dists = np.zeros((indices.shape[0]))
    preds = np.zeros((indices.shape[0]))
    while True:
        next_i = [(j,indices[j,ii],ii) for j in prev_i for ii in range(indices[j,:].shape[0])]
        for ind in next_i:
            min_ind = None
            min_min = None
            if sptset[ind[1]] == 0.0:
                if min_ind == None:
                    min_ind = ind[1]
                    min_min = dists[ind[0]]+distances[ind[0],ind[2]]
                    preds[ind[1]] = ind[0]
                elif min_min < (dists[ind[0]]+distances[ind[0],ind[2]]):
                    min_ind = ind[1]
                    min_min = dists[ind[0]]+distances[ind[0],ind[2]]
                    preds[ind[1]] = ind[0]
        sptset[min_ind] = 1.0
        dists[min_ind] = min_min
        prev_i = [j for j in range(distances.shape[0]) if sptset[j] == 1.0]
        if min_ind == v:
            break
        if sum(sptset) == indices.shape[0]:
            break
    return dists,preds

def shortest_path(start,end,distances,indices):
    dists,preds = dijkstra(start,end,distances,indices)
    v = end
    path = [int(v)]
    while v != start:
        v = preds[int(v)]
        path.append(int(v))
    path.reverse()
    return path,len(path)

def sequence_path(args,data,k):
    start,end = args
    X = np.concatenate([data.reshape((data.shape[0],-1)),
                        start.reshape((1,-1)),
                        end.reshape((1,-1))],axis=0)
    bt = BallTree(X,leaf_size=3,metric='hamming')
    
    distances,indices = bt.query(X,k=k,return_distance=True)
    path,nsteps = shortest_path(X.shape[0]-2,X.shape[0]-1,distances,indices)
    

    return [X[p,:].reshape(data.shape[1:]) for p in path],nsteps
    
                                        
    
    
