# Parametric bounds


## Affine bounds

### Affine layer with constant input bounds

- constant input bounds $[x_{\ell}, x_u]$
- modifiable parameters $\dot{W}, \dot{b}$
- variable output bounds $[\dot{y}_{\ell}, \dot{y}_u]$

#### Upper bound
$$
\dot{y}_{u,j} = \sum_i \dot{V}_{u, ji} + \dot{b}_j \land \dot{V}_{u} \geq \dot{W} x_{\ell} \land \dot{V}_{u} \geq \dot{W} x_u
$$

#### Lower bound
$$
\dot{y}_{\ell,j} = \sum_i \dot{V}_{\ell, ji} + \dot{b}_j \land \dot{V}_{\ell} \leq \dot{W} x_{\ell} \land \dot{V}_{\ell} \leq \dot{W} x_u
$$

### Affine layer with variable input bounds

- variable input bounds $[\dot{x}_{\ell}, \dot{x}_u]$
- fixed parameters $W$
- modifiable parameters $\dot{b}$
- variable output bounds $[\dot{y}_{\ell}, \dot{y}_u]$

$$
\dot{y}_{u,j} = \sum_i V_{u, ji} + \dot{b}_j \\
\dot{y}_{\ell,j} = \sum_i V_{\ell, ji} + \dot{b}_j \\
$$
where
$$
V_{u, ji} = \dot{x}_{u,i} W_{ji}, \text{ and } V_{\ell, ji} = \dot{x}_{\ell,i} W_{ji} \text{ if } W_{ji} \geq 0 \\
V_{u, ji} = \dot{x}_{\ell,i} W_{ji}, \text{ and } V_{\ell, ji} = \dot{x}_{u,i} W_{ji} \text{ otherwise}
$$


## ReLU bounds

- variable input bounds $[\dot{x}_{\ell}, \dot{x}_u]$
- variable output bounds $[\dot{y}_{\ell}, \dot{y}_u]$

$$
\dot{y}_{u} \geq 0 \land \dot{y}_{u} \geq \dot{x}_u \\
\dot{y}_{\ell} = \alpha \dot{x}_{\ell}
$$
where $\alpha \in [0, 1]$ is the slope of the lower bound.
