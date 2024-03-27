import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
print(batch)
OPTS = []

if batch == '':
    OUT_FILE = 'args/base.txt'
    opts = {'manifold': ['PoincareBall'], 'w_ms': [1e-4]} 
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean']} 
    OPTS.append(opts)
    opts = {'w_pde':[1e-20], 'w_gpde': [1e-20]}
    OPTS.append(opts)
    opts = {'decoder':['ResNet']}
    OPTS.append(opts)

elif batch == 'scaling':
    OUT_FILE = 'args/scaling.txt'
    opts = {'manifold': ['PoincareBall'], 'path': ['t1708806913']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'level_emb_var': [1], 'path': ['t1708806913']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'decoder': ['ResNet'], 'path': ['t1708806913']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean'], 'path': ['t1708806913']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'w_pde':[0.], 'w_gpde': [0.], 'path': ['t1708806913']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)

elif batch == 'scaling_2':
    opts = {'manifold': ['PoincareBall'], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'level_emb_var': [1], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'decoder': ['ResNet'], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'branch_net': ['ResNet'], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean'], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)
    opts = {'w_pde':[0.], 'w_gpde': [0.], 'path': ['t1708805081']} #, 't1708806913', 't1708808123']}
    OPTS.append(opts)

elif batch == 'uncertainty':
    OUT_FILE = 'args/uncertainty.txt'
    opts = {'manifold': ['PoincareBall'], 'eta_var': [1.0e-04, 2.5e-03, 1.0e-02]}
    OPTS.append(opts)
    opts = {'decoder': ['ResNet'], 'eta_var': [1.0e-04, 2.5e-03, 1.0e-02]}
    OPTS.append(opts)
    opts = {'w_pde':[0.], 'w_gpde': [0.], 'eta_var': [1.0e-04, 2.5e-03, 1.0e-02]}
    OPTS.append(opts)
    opts = {'manifold': ['Euclidean'], 'eta_var': [1.0e-04, 2.5e-03, 1.0e-02]}
    OPTS.append(opts)

elif batch == 'enc':
    OUT_FILE = 'args/enc.txt'
    opts = {'c':[1/4, 1], 'epochs':[100000]}

elif batch == 'embed':
    OUT_FILE = 'args/embed.txt'
    opts = {'pos_emb_var': [0, 1], 'level_emb_var': [0, 1]}
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
