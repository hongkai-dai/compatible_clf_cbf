# Maximizing ellipsoid volume

Say we have a semi-algebraic set $S = \{x\in\mathbb{R}^n | g(x) < 0 \}$ and we want to find a large inscribed ellipsoid $\mathcal{E}\subset \mathcal{S}$.

## Formulation 1

Consider to parameterize the ellipsoid as $\mathcal{E} = \{x | x^TSx + b^Tx + c \le 0\}$. The condition that $\mathcal{E}\subset\mathcal{S}$ is

$$
\begin{align}
-1 - \phi_0(x)g(x) + \phi_1(x)(x^TSx+b^Tx+c) \text{ is sos}\\
\phi_0(x), \phi_1(x) \text{ are sos}.
\end{align}
$$

The volume of the ellipsoid is proportional to

$$
vol(\mathcal{E})\propto \left(\frac{b^TS^{-1}b/4-c}{\text{det}(S)^{1/n}}\right)^{\frac{n}{2}}
$$

So to maximize the ellipsoid, we can maximize 

$$
\begin{align}
\frac{b^TS^{-1}b/4-c}{\text{det}(S)^{1/n}}
\end{align}
$$

I don’t think we can maximize this term directly through convex optimization, so we will seek to maximize a surrogate function instead, which can be done through convex optimization.

Since logarithm function is monotonically increasing, we can maximize the log of Eq 3

$$
\begin{align}
\max_{S, b, c} \log(b^TS^{-1}b/4-c) - \frac{1}{n}\log\text{det}(S)
\end{align}
$$

This objective (4) is still non-convex. So we consider to linearize this objective and maximize the linearization. To compute the linearized objective, by Schur complement, we know that

$$
(c - b^TS^{-1}b/4) \bullet \text{det}(S) = \text{det}\left(\begin{bmatrix}c & b^T/2\\b/2 & S\end{bmatrix}\right)
$$

Therefore we have 

$$
\log(b^TS^{-1}b/4-c) = \log\left(-\text{det}\begin{bmatrix}c & b^T/2 \\b/2 & S\end{bmatrix}\right) - \log \text{det}(S)
$$

Hence the objective function (4) can be re-formulated as

$$
\begin{align}
\max_{S, b, c} \log \left(-\text{det}\left(\begin{bmatrix}c & b^T/2 \\b/2 & S\end{bmatrix}\right)\right) - (1+\frac{1}{n})\log\text{det}(S)
\end{align}
$$

It is easier to compute the linearization of objective (5) than objective (4), by using the following property on the gradient of $\log \text{det}(X)$ for a symmetric matrix $X$:m

$$
\begin{align}
\frac{\partial\log\text{det}(X)}{\partial X} = X^{-1}\text{ if } \text{det}(X) > 0\\
\frac{\partial\log(-\text{det}(X))}{\partial X} = X^{-1}\text{ if } \text{det}(X) < 0
\end{align}
$$

So the linearization of the objective in (5) at a point $(S^{(i)}, b^{(i)}, c^{(i)})$ is

$$
\left<\begin{bmatrix}c & b^T/2\\b/2 & S\end{bmatrix}, \begin{bmatrix}c^{(i)} &(b^{(i)})^T/2\\b^{(i)}/2 & S^{(i)}\end{bmatrix}^{-1}\right> - (1+\frac{1}{n})\left<S, (S^{(i)})^{-1}\right>
$$

where we use $<X, Y> \equiv \text{trace}(X^TY)$ to denote matrix inner product.

Furthermore, to ensure that the $\{x | x^TSx+b^Tx+c\le 0\}$ really represents an ellipsoid. We need to enforce the constraint $b^TS^{-1}b/4-c \ge 0$. Unfortunately this constraint is non-convex. To remedy this, we consider a sufficient condition, that a given point $\bar{x}$ is inside the ellipsoid, namely

$$
\bar{x}^TS\bar{x}+b^T\bar{x}+c\le 0
$$

as a linear constraint on $(S, b, c)$.

So to summarize, we solve a sequence of convex optimization problem

$$
\begin{align}
\max_{S, b, c, \phi_0} &\left<\begin{bmatrix}c & b^T/2\\b/2 & S\end{bmatrix}, \begin{bmatrix}c^{(i)} &(b^{(i)})^T/2\\b^{(i)}/2 & S^{(i)}\end{bmatrix}^{-1}\right> - (1+\frac{1}{n})\left<S, (S^{(i)})^{-1}\right>\\
\text{s.t } &\bar{x}^TS\bar{x} + b^T\bar{x} + c \le 0\\
&-1 - \phi_0(x)g(x) + \phi_1(x)(x^TSx+b^Tx+c) \text{ is sos}\\
&\phi_0(x), \phi_1(x) \text{ are sos}.\end{align}
$$

In the i’th iteration we get a solution $(S^{(i)}, b^{(i)}, c^{(i)})$, we linearize the objective (5) at this solution and maximize the linear objective again.

## Formulation 2

We consider an alternative parameterization of the ellipsoid as $\mathcal{E}=\{x | \Vert Ax+b\Vert_2\le 1\}, A\succ 0$. To maximize the volume of this ellipsoid, we need to minimize $\log \text{det}(A)$. But $\log \text{det}(A)$ is a concave function. To minimize this concave function, we again use the sequential linearization idea as in formulation 1. Namely we linearize this objective at a point $A^{(i)}$, and minimize the linearized objective

$$
\left<A - A^{(i)}, (A^{(i)})^{-1}\right>
$$

Moreover, to impose the constraint that $\mathcal{E}\subset\mathcal{S}$, we first use the Schur complement to convert $\Vert Ax+b\Vert_2\le 1$ to a linear constraint on $(A, b)$

$$
\Vert Ax+b\Vert_2\le 1 \Leftrightarrow \begin{bmatrix} 1 &(Ax+b)^T\\Ax+b & I\end{bmatrix}\succeq 0
$$

Hence a sufficient condition for $\mathcal{E}\subset\mathcal{S}$ is

$$
\begin{align}
g(x)\Phi_0(x) - \phi_1(x)\begin{bmatrix}1 & (Ax+b)^T\\Ax+b & I \end{bmatrix} \text{ is psd}\\
\Phi_0(x) \text{ is psd}, \phi_1(x) \text{ is sos}
\end{align}
$$

Notice that this condition is significantly more complicated than the p-satz condition (Eq (1) and (2)) in Formulation 1, as (12) (13) require matrix-sos conditions.