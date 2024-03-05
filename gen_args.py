import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
print(batch)
OPTS = []

if batch == '':
    OUT_FILE = 'args/base.txt'
    opts = {'manifold': ['Euclidean','PoincareBall'], 'w_pde':[1e-20, 1e+0], 'w_gpde': [1e-20, 1e+3] }
    OPTS.append(opts)

elif batch == 'optim':
    OUT_FILE = 'args/optim.txt'
    opts = {'weight_decay': [1e-2], 'b1': [.9], 'epochs': [5000], 'log_path': [1709607155]}
    OPTS.append(opts)
    opts = {'weight_decay': [1e-3], 'b1': [.99], 'epochs': [5000],'log_path': [1709607155]}
    OPTS.append(opts)
    opts = {'use_layer_norm': [1], 'log_path': [1709607155]}
    OPTS.append(opts)


elif batch == 'clip':
    OUT_FILE = 'args/clip.txt'
    opts = {'max_norm': [.01,.005], 'max_norm_enc': [.001,.005], 'w_ent': [1e-3,1e-4]} 
    OPTS.append(opts)
    opts = {'t_var': [.01,.005], 'x_var': [.001,.0005]} 
    OPTS.append(opts)

elif batch == 'agg':
    OUT_FILE = 'args/agg.txt'
    opts = {'agg': ['multi', 'mean', 'sum']}
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean','PoincareBall']}
    OPTS.append(opts)
    opts = {'w_pde':[1e-20, 1e+0], 'w_gpde': [1e-20, 1e+3]}
    OPTS.append(opts)

elif batch == 'adam':
    OUT_FILE = 'args/adam.txt'
    opts = {'b1': [.9,.95,.999], 'weight_decay': [1e-3, 2e-3], 'epochs': [50000]} 

else:
    print('argv[1] not recognized!')
    raise

print(f'out file: {OUT_FILE}')

if __name__ == '__main__':
    
    ### take the cartesian product of all opts and write 
    ### arg strings (e.g. --model MLP --dropout 0.6 ...) to file
    #OUT_FILE = sys.argv[1] if len(sys.argv)>1 else 'args.txt'
    
    if len(OPTS)==0:
        vals = list(itertools.product(*opts.values()))
        args = [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    elif len(OPTS)>0:
        vals = []
        args = []
        for opts in OPTS: 
            vals = list(itertools.product(*opts.values()))
            args += [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    with open(OUT_FILE,'w') as fp: fp.write('\n'.join(args))
