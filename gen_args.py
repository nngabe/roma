import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
print(batch)
OPTS = []

if batch == '':
    OUT_FILE = 'args/base.txt'
    opts = {'manifold': ['Poincare']} 
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean']} 
    OPTS.append(opts)
    opts = {'w_pde':[1e-20], 'w_gpde': [1e-20]}
    OPTS.append(opts)
    opts = {'decoder':['ResNet']}
    OPTS.append(opts)

elif batch == 'test':
    OUT_FILE = 'args/test.txt'
    opts = {'path': [1716416941], 'epochs': [200000]} # 38k 
    OPTS.append(opts)
    opts = {'path': [1715101625], 'epochs': [200000]} # 1M 
    OPTS.append(opts)
    opts = {'path': [11716416941], 'eta_var': [0.01]} # 38k, 10% noise 
    OPTS.append(opts)
    opts = {'path': [1718042027]} # 38k, non-stationary 
    OPTS.append(opts)
    opts = {'path': [1715275273]} # 29k, Burgers 
    OPTS.append(opts)

elif batch.isnumeric():
    opt_ = {'path': [batch], 'eta_var': [1e-4]}
    name = '_'.join([i[0]+str(i[1]) for i in opt_.items()])
    OUT_FILE = f'args/ablations_{name}.txt'
   
    # default ROMA
    opts = opt_ 
    OPTS.append(opts)
   
    # PINN decoder
    opts = opt_ | {'decoder': ['ResNet'], 'dec_width': [960]}
    OPTS.append(opts)
   
    # no renorm 
    opts = opt_ | {'pool_steps': [0]}
    OPTS.append(opts)
   
    # Euclidean manifold 
    #opts = opt_ | {'manifold': ['Euclidean']}
    #OPTS.append(opts)
    
    # no PDE/gPDE
    opts = opt_ | {'w_pde':[0.], 'w_gpde': [0.]}
    OPTS.append(opts)
    
    # no graph positional encoding
    opts = opt_ | {'pe_dim': [0]}
    OPTS.append(opts)

elif batch == 'ablations_extra':
    OUT_FILE = 'args/ablations.txt'
    opts = {'w_gpde': [0.]}
    OPTS.append(opts)
    opts = {'level_emb_var': [1]}
    OPTS.append(opts)
    opts = {'coord_dim': [1]}
    OPTS.append(opts)
    opts = {'w_gpde': [0.]}
    OPTS.append(opts)
    opts = {'w_pool': [1,2,3]}
    OPTS.append(opts)

elif batch == 'walk':
    OUT_FILE = 'args/walk.txt'
    opt_ = {'path': [1716237798], 'epochs': [25000], 'lr': [1e-6]}

    opts = opt_ | {'max_walk_len': [12,16,24]}
    OPTS.append(opts)

elif batch == 'optim':
    OUT_FILE = 'args/optim.txt'
    opts = {'weight_decay': [1e-2], 'b1': [.9], 'epochs': [10000], 'log_path': [1709754619]}
    OPTS.append(opts)
    opts = {'weight_decay': [1e-3], 'b1': [.9], 'epochs': [10000],'log_path': [1709754619]}
    OPTS.append(opts)
    opts = {'epsilon': [1e-4] ,'weight_decay': [1e-3], 'b1': [.9], 'epochs': [10000], 'log_path': [1709754619]}
    OPTS.append(opts)
    opts = {'optim': ['adamw'], 'weight_decay': [1e-3], 'b1': [.9], 'epochs': [10000], 'log_path': [1709754619]}
    OPTS.append(opts)

elif batch == 'ffe':
    OUT_FILE = 'args/ffe.txt'
    opts = {'t_var': [1e-4, 1e-6], 'x_var': [1e-4, 1e-6]} 
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

elif batch == 'adam':
    OUT_FILE = 'args/adam.txt'
    opts = {'b1': [.9,.95,.999], 'weight_decay': [1e-3, 2e-3], 'epochs': [50000]} 

else:
    print('argv[1] not recognized!')
    raise

print(f'out file: {OUT_FILE}')

if __name__ == '__main__':
    
    ### take the cartesian product of all opts and write arg
    ### strings (e.g. --model MLP --dropout 0.6 ...) to file given by:
    ### OUT_FILE = sys.argv[1] if len(sys.argv)>1 else 'args.txt'
    
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
