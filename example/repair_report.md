# Repair execution report

This report can be reproduced as a detailed numerical trace with:

```bash
uv run python example/track_repair.py
```

## Configuration

| Item | Value |
|---|---|
| Input region | $[3,8]\times[0,8]$ |
| Property | $o\geq3$ |
| Per-weight modification limit | $0.5$ |
| Original $W_1$ | $[[2,-1],[-2,1]]$ |
| Original $W_2$ | $[3,-3]$ |

## Initial verification

$$
x^{(1)}\in[-2,16]\times[-16,2],\qquad
\hat{x}^{(1)}\in[0,16]\times[0,2],\qquad o\in[-6,48].
$$

Both ReLUs are unstable. Their convex upper slopes are respectively $8/9$
and $1/9$.

**Result:** failed, because the output lower bound is $-6<3$.

## Single-layer repair result

Only $W_2$ is modifiable. Points with $2x^{(0)}_1=x^{(0)}_2$ make both hidden
activations zero, so last-layer weights alone cannot establish $o\geq3$.
The lower-bound objective selects the second coefficient as $-2.5$; the first
coefficient is irrelevant to the lower bound, so it is retained as $3$.

$$
W'_2=[3,-2.5],\qquad o'\in[-5,48].
$$

**Result:** failed, because the best achievable lower bound is $-5<3$.

## Multilayer repair trace

### Safe input search

For

$$
Q^{(0)}(r)=[5.5-2.5r,5.5+2.5r]\times[4-4r,4+4r],
$$

the certified output interval is $[21-27r,21+27r]$. Binary search
converges to

$$
r^*=\frac23\approx0.666667,
$$

and returns

$$
Q^{(0)}=\left[\frac{23}{6},\frac{43}{6}\right]
\times\left[\frac43,\frac{20}{3}\right],
$$

with $o\in[3,39]$.

### Intermediate safe-subspace search

Propagation through $W_1$ gives $Q^{(0,1)}=[1,13]\times[-13,-1]$.
Doubling its radius produces

$$
[-5,19]\times[-19,5],
$$

whose output lower bound is $-15$. This exponential candidate is rejected.
Binary refinement returns expansion scale $1$, so the intermediate box remains
$[1,13]\times[-13,-1]$.

### First-layer subrepair

The bound-violation objective selects

$$
W'_1=\begin{bmatrix}13/8&-1/2\\-13/8&1/2\end{bmatrix}.
$$

| Quantity | Before repair | After first-layer repair |
|---|---:|---:|
| $x^{(1)}_1$ | $[-2,16]$ | $[7/8,13]$ |
| $x^{(1)}_2$ | $[-16,2]$ | $[-13,-7/8]$ |
| $\hat{x}^{(1)}_1$ | $[0,16]$ | $[7/8,13]$ |
| $\hat{x}^{(1)}_2$ | $[0,2]$ | $[0,0]$ |
| $o$ with original $W_2$ | $[-6,48]$ | $[21/8,39]$ |

The repair stabilizes both ReLUs and improves the lower output bound to $21/8$,
but the property still fails.

### Last-layer subrepair

Using $\hat{x}^{(1)}\in[7/8,13]\times\{0\}$, maximizing the output lower bound
selects the first output weight as $3.5$. A representative solution is

$$
W'_2=[3.5,-3],\qquad o'\in[49/16,91/2].
$$

**Result:** successful. The repaired network satisfies $o'\geq3$ for every
$x^{(0)}\in[3,8]\times[0,8]$.

## Summary

| Method/stage | Lower bound | Upper bound | Outcome |
|---|---:|---:|---|
| Original network | $-6$ | $48$ | Failure |
| Single-layer repair | $-5$ | $48$ | Failure |
| After first multilayer subrepair | $21/8$ | $39$ | Failure |
| Completed multilayer repair | $49/16$ | $91/2$ | Success |
