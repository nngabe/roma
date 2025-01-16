import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
ablation = sys.argv[2] if len(sys.argv)>2 else False
opt1 = int(sys.argv[3]) if len(sys.argv)>3 else 1
opt2 = int(sys.argv[4]) if len(sys.argv)>4 else 1

print(batch)
OPTS = []


dim_size = {'1718042027': 512, '1722538024': 256, '1725828780': 512, '1722285916': 512}

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
    opts = {'path': [1718042027]} # 38k 
    OPTS.append(opts)
    opts = {'path': [1722538024]} # 314k 
    OPTS.append(opts)
    opts = {'path': [1725828780]} # 2M Burgers 
    OPTS.append(opts)
    opts = {'path': [1722285916]} # 3M Kuramoto
    OPTS.append(opts)

elif batch.isnumeric() and not ablation:
    opt_ = {'path': [batch], 'pe_dim': [dim_size[batch]]}
    name = '_'.join([i[0]+str(i[1][0])[-4:] for i in opt_.items()])
    OUT_FILE = f'args/base_{name}.txt'
   
    # ROMA
    opts = opt_ 
    OPTS.append(opts)
 
    # DON
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'enc_depth': [0], 'nonlinear': [0], 'branch_net': ['Res'], 'pe_embed_dim': [0], 'dec_width': [1872]} 
    OPTS.append(opts)
    
    # DON-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'enc_depth': [1], 'nonlinear': [0], 
                   'branch_net': ['Transformer'], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'edge_conv': [1],
                    'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1326]}
    OPTS.append(opts)
    
    # NOMAD-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'enc_depth': [1], 'nonlinear': [1],
                   'branch_net': ['Transformer'], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'edge_conv': [1], 
                    'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1292]}
    OPTS.append(opts) 




elif batch.isnumeric() and ablation=='BD':
    opt_ = {'path': [batch], 'pde': ['burgers'], 'x_var': [1e-3], 't_var': [1e-3], 'lr': [5e-6], 'steps': [80000]}
    name = '_'.join([[i[0]+str(i[1][0])[-4:] for i in opt_.items()][0]])
    OUT_FILE = f'args/ed_{name}.txt'

    # ROMA
    opts = opt_ 
    OPTS.append(opts)
 
    # DON-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'enc_depth': [0], 'nonlinear': [0], 'branch_net': ['Res'], 'pe_embed_dim': [0], 'dec_width': [1482]} 
    OPTS.append(opts)
    
    # DON-MP-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'nonlinear': [0], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 
                   'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1072]}
    OPTS.append(opts)
    
    # NOMAD-MP-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'nonlinear': [1], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 
                   'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1050]}
    OPTS.append(opts) 
 
elif batch.isnumeric() and ablation=='KM':
    opt_ = {'path': [batch]}
    name = '_'.join([i[0]+str(i[1][0])[-4:] for i in opt_.items()])
    OUT_FILE = f'args/ed_{name}.txt'

    # ROMA
    opts = opt_ 
    OPTS.append(opts)
 
    # DON-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'enc_depth': [0], 'nonlinear': [0], 'branch_net': ['Res'], 'pe_embed_dim': [0], 'dec_width': [1524]}
    OPTS.append(opts)
    
    # DON-MP-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'nonlinear': [0], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 
                   'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1080]}
    OPTS.append(opts)
    
    # NOMAD-MP-PI
    opts = opt_ | {'pool_steps': [0], 'w_pde': [1.], 'w_gpde': [1.], 'nonlinear': [1], 'pe_embed_dim': [0], 'manifold': ['Euclidean'], 
                   'func_pos_emb': [0], 'dual_pos_emb': [5], 'dec_width': [1054]}
    OPTS.append(opts) 
 
elif batch.isnumeric() and ablation == 'PE':
    opt_ = {'path': [batch], 'func_pos_emb': [opt1], 'dual_pos_emb': [opt2]}
    OUT_FILE = f'args/pe_{opt_["path"][0]}_dual{opt_["func_pos_emb"][0]}.{opt_["dual_pos_emb"][0]}.txt'

    opts = opt_ | {'pos_emb_var': [0., .2, .4, .6, .8, 1.]}
    OPTS.append(opts)


elif batch.isnumeric() and ablation == 'HN':
    opt_ = {'path': [batch], 'eta_var': [0.01]}
    name = '_'.join([i[0]+str(i[1][0])[-4:] for i in opt_.items()])
    OUT_FILE = f'args/hn_{name}.txt'

    # ROMA
    opts = opt_ 
    OPTS.append(opts)
 
    # DON
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'enc_depth': [0], 'nonlinear': [0], 'branch_net': ['Res'], 'pe_embed_dim': [0]} 
    OPTS.append(opts)
    
    # DON-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'nonlinear': [0], 
                  'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'func_pos_emb': [0], 'dual_pos_emb': [5]}
    OPTS.append(opts)
    
    # NOMAD-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'nonlinear': [1],
                   'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'func_pos_emb': [0], 'dual_pos_emb': [5]}
    OPTS.append(opts) 
 
elif batch.isnumeric() and ablation == 'UQ':
    opt_ = {'path': [batch]}
    OUT_FILE = f'args/uq_{opt_["path"][0]}.txt'

    opt_ = opt_ | {'eta_var': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]} 
    
     # ROMA
    opts = opt_ 
    OPTS.append(opts)
 
    # DON
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'enc_depth': [0], 'nonlinear': [0], 'branch_net': ['Res'], 'pe_embed_dim': [0]} 
    OPTS.append(opts)
    
    # DON-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'nonlinear': [0], 
                  'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'func_pos_emb': [0], 'dual_pos_emb': [5]}
    OPTS.append(opts)
    
    # NOMAD-MP
    opts = opt_ | {'pool_steps': [0], 'w_pde': [0.], 'w_gpde': [0.], 'nonlinear': [1],
                   'pe_embed_dim': [0], 'manifold': ['Euclidean'], 'func_pos_emb': [0], 'dual_pos_emb': [5]}
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
    opts = opt_ | {'pe_dim': [0]}
    OPTS.append(opts)

elif batch == 'walk':
    OUT_FILE = 'args/walk.txt'
    opt_ = {'path': [1716237798], 'epochs': [25000], 'lr': [1e-6]}

    opts = opt_ | {'max_walk_len': [12,16,24]}
    OPTS.append(opts)

elif batch == 'ffe':
    OUT_FILE = 'args/ffe.txt'
    opt_ = {}
    opts = opt_ | {'t_var': [1e-5], 'x_var': [1e-5]} 
    OPTS.append(opts)
    opts = opt_ | {'t_var': [1e-6], 'x_var': [1e-6]} 
    OPTS.append(opts)
    opts = opt_ | {'t_var': [1e-4], 'x_var': [1e-3]} 
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
