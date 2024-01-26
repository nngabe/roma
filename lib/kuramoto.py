import warnings
warnings.filterwarnings('ignore')

## plotting preferences
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,6]; plt.rcParams['font.size'] = 24; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True

import argparse
import time
import pandas as pd
import networkx as nx
import numpy as np
import glob
import os,sys
from collections import defaultdict

from community import community_louvain
import jax
import jax.numpy as jnp
from jax.experimental import ode
import jraph
import scipy

from graph_utils import gen_graph

np.random.seed(123)
PATH = os.path.expanduser('~/data')

parser = argparse.ArgumentParser()
parser.add_argument('--precision', nargs='?', default=1e-16, type=float)
parser.add_argument('--N', nargs='?', default=10000, type=int)
parser.add_argument('--c', nargs='?', default=1., type=float)
parser.add_argument('--K', nargs='?', default=5., type=float)
parser.add_argument('--a', nargs='?', default=.5, type=float)
parser.add_argument('--b', nargs='?', default=10., type=float)
parser.add_argument('--g', nargs='?', default=10., type=float)
parser.add_argument('--d', nargs='?', default=.5, type=float)
parser.add_argument('--steps', nargs='?', default=20000, type=int)
parser.add_argument('--dt', nargs='?', default=1e-3, type=float)
parser.add_argument('--save', nargs='?', default=0, type=int)
parser.add_argument('--seed', nargs='?', default=123, type=int)
parser.add_argument('--plot', nargs='?', default=False, type=bool)
parser.add_argument('--gplot', nargs='?', default=False, type=bool)
parser.add_argument('--periods', nargs='?', default=1., type=float)
args = parser.parse_args()

tic = time.time()

avg_degree = 3.
c=args.c
N=args.N
seed=71

G, part, re = gen_graph(N, avg_degree, c, seed=seed, plot=args.gplot)
from jax.config import config; config.update("jax_enable_x64", True)

com = defaultdict(list)
for key, value in part.items():
    com[value].append(re[key])
    
edge_list = np.array(list(G.edges)).T
edge_file = f'{PATH}/edges_c{c}_N{G.number_of_nodes()}_t{int(tic)}.csv'
pd.DataFrame(edge_list).to_csv(edge_file)

print(f' N = {G.number_of_nodes()}')
print(f'|E| = {G.number_of_edges()}')

A = np.array(nx.adjacency_matrix(G).todense())
s,r = np.where(A)
N = A.shape[0]

sigma = 1.
omega_i =  np.clip(np.random.normal(.0, sigma, N), -5*sigma, 5*sigma)
omega_i -= omega_i.mean()
print(f'E[omega] = {omega_i.mean():.3f}, Var[omega] = {omega_i.std():.3f}')

K = args.K
eps = .00
dt = args.dt
steps = args.steps
p=args.precision

x = np.zeros((N,steps+1))
x[:,0] =  5. * np.random.normal(0.,1.,N)

w = (A.sum(1)/A.shape[0])**.5
norm = jraph.segment_sum(w[s], r, N)
w_ij = w[s]/norm[r]

def _K(t, K=args.K, a=args.a, b=args.b, g=args.g, d=args.d):
    t *= jnp.pi/(dt*steps)*args.periods
    res = ((1-d)*jnp.abs(jnp.cos(g*t))**b + d) * jnp.abs(jnp.sin(t))**2. * (1.-a) + a
    return K * res

@jax.jit
def f(x,t):
    m_ij = jnp.sin(x[s] - x[r])
    msg_i = _K(t) * jraph.segment_sum(m_ij*w_ij, r, N)
    return omega_i + msg_i

tt = np.linspace(0.,dt*steps,steps)
Ktt = _K(tt)
if args.plot:
    pd.DataFrame(Ktt,columns=[r'$K(t)$']).plot()

key = jax.random.PRNGKey(0)

x = ode.odeint(f, x[:,0], dt*jnp.arange(0,steps), rtol=p, atol=p)
print(f'compute time (jax DP): {time.time()-tic:.3f} (sec)'); tic = time.time()
x = np.array(x).T


pd.DataFrame(np.where(A)).to_csv(f'{PATH}/adj_K{K}.{10.*Ktt.var():.2f}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.csv')
pd.DataFrame(x).to_csv(f'{PATH}/x_K{K}.{10.*Ktt.var():.2f}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.csv')

if not args.plot: sys.exit(0)

ax = pd.DataFrame(x.T).diff().iloc[steps//10::100,:40].plot(legend=False,logx=False)
ax.set_ylabel(r'$\theta_i$', rotation=90, fontsize=30); ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
plt.tight_layout()
plt.savefig(f'{PATH}/kur_K{K}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.pdf')

def _R(x,i=None):
    x = x if i==None else x[i]
    R = np.exp(1j*x).mean(0)
    return np.abs(R)

colors = ['purple', 'forestgreen', 'royalblue', 'crimson']*2
R = _R(x)
ax = pd.DataFrame(R,columns=[r'$R(t)$']).plot(logx=True,logy=True,legend=True,ylim=[min(1e-2,.9*min(R)), 1.], style='r--')
ax.set_ylabel(r'$R(t)$', rotation=90, fontsize=30,labelpad=5) ;ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
Rc = np.array([_R(x,c) for c in com.values()])
pd.DataFrame(Rc.mean(0),columns=[r'$R_c(t)$']).plot(ax=ax, logx=False, logy=True, legend=True, ylim=[min(1e-2,.9*min(R)), 1.], style='b--')
pd.DataFrame(np.ones_like(R)/np.sqrt(N),columns=[r'$1/\sqrt{N}$']).plot(ax=ax,style='k-.')
for j in range(0):
    pd.DataFrame(Rc[j]).plot(ax=ax, logx=False, logy=True, legend=False, ylim=[min(1e-2,.9*min(R)), 1.], color = colors[j])
    ax.set_ylabel(r'$R(t)$', rotation=90, fontsize=30,labelpad=5) ;ax.set_xlabel(r'$t$', rotation=0, fontsize=30)

plt.tight_layout()
plt.savefig(f'{PATH}/order_K{K}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.pdf')
print(f' R = {R.mean()} \u00b1 {R.std()}')
print(f' Rc/R = {(Rc.mean(0)/R)[1000:].mean()}')
dx = pd.DataFrame(x).diff(axis=1).dropna(axis=1).to_numpy()
dx = dx-dx.mean(1).reshape(-1,1)
dx = dx[:,dx.shape[1]//5:]
