# Minimizing ellipsoid volume

Say we have a semi-algebraic set $\mathcal{S} = \{x \in\mathbb{R}^n| g(x) \le 0\}$, we want to find the smallest ellipsoid containing this set.

## Formulation 1

Let’s consider the ellipsoid parameterized as $\mathcal{E}=\{x | x^TSx+b^Tx+c< 0\}$. The constraint that the ellipsoid $\mathcal{E}=\{x | x^TSx+b^Tx+c < 0\} \supset \mathcal{S}=\{x | g(x)\le 0\}$ can be imposed through the *positivstellensatz* (p-satz)

$$
\begin{align}
-1-\phi_0(x)(x^TSx+b^Tx+c) +\phi_1(x)g(x) \text{ is sos}\\
\phi_0(x), \phi_1(x) \text{ are sos}
\end{align}
$$

The volume of this ellipsoid is proportional to

$$
vol(\mathcal{E})\propto \left(\frac{b^TS^{-1}b/4-c}{\text{det}(S)^{1/n}}\right)^{\frac{n}{2}}
$$

Minimizing this volume is equivalent to minimizing

$$
\begin{align}
\frac{b^TS^{-1}b/4-c}{\text{det}(S)^{1/n}}
\end{align}
$$

How to minimize (3) through convex optimization? Here we try several attempts

### Attempt 1

 Taking the logarithm of (3) we get

$$
\begin{align}
\log(b^TS^{-1}b/4-c) - \frac{1}{n}\log\text{det}(S)
\end{align}
$$

First note that $\log\text{det}(S)$ is a concave function, hence we can minimize $-\frac{1}{n}\log\text{det}(S)$ through convex optimization.

Second we notice that we can minimize $b^TS^{-1}b/4-c$ through convex optimization. By using Schur complement, we have $b^TS^{-1}b/4-c\le r$ if and only if the following matrix is psd

$$
\begin{align}
\begin{bmatrix} c+r & b^T/2\\b/2 & S\end{bmatrix} \succeq 0
\end{align}
$$

Hence we can minimize $r$ subject to the convex constraint (5).

Unfortunately we cannot minimize $\log r$ through convex optimization (it is a concave function of $r$). Hence this attempt isn’t successful.

### Attempt 2

Let’s try again. We consider the following optimization program

$$
\begin{align}
\min_{S, b, c} b^TS^{-1}b/4-c\\
\text{s.t } \text{det}(S) \ge 1
\end{align}
$$

Note that the constraint (7) is equivalent to

$$
\begin{align}
\log \text{det}(S) \ge 0
\end{align}
$$

which is a convex constraint. Hence we can solve the objective (6) subject to the constraint (8) through the convex optimization problem

$$
\begin{align}
\min_{S, b, c, r} &\;r\\
\text{s.t }& \begin{bmatrix}c+r & b^T/2\\b/2 & S\end{bmatrix}\succeq 0\\
&\log\text{det}(S) \ge 0
\end{align}
$$

Is this optimization problem (9)-(11) (which is equivalent to (6)(7)) same as minimizing the original objective (3)? First we notice that the optimal cost in (6)(7) is an upper bound of the minimization over (3), because we constrain the denominator $\text{det}(S)\ge 1$.  Second, if the optimal solution to minimizing (3) is $(\bar{S}, \bar{b}, \bar{c})$, then we can construct $(S, b, c) = (\bar{S}, \bar{b}, \bar{c}) / \text{det}(\bar{S})^{\frac{1}{n}}$. This newly constructed $(S, b, c)$ satisfies constraint (7), and the cost of this $(S, b, c)$ in (6) is exactly the cost of $(\bar{S}, \bar{b}, \bar{c})$ in (3). Hence the optimization problem (6)(7) achieves the same cost as minimizing the original objective (3).

## Formulation 2

We consider an alternative formulation on the ellipsoid $\mathcal{E} = \{x | \Vert Ax+b\Vert_2< 1\}$ with $A\succeq 0$. Minimizing the volume of this ellipsoid is equivalent to maximizing $\log\text{det}(A)$. 

To imposing that $\mathcal{E}\supset\mathcal{S}$, we first consider the following relationship from Schur complement

$$
\begin{align}
\Vert Ax+b\Vert_2 < 1 \Leftrightarrow &\begin{bmatrix}1 & (Ax+b)^T\\Ax+b & I\end{bmatrix}\succ 0 \\\Leftrightarrow & \begin{bmatrix}1\\y\end{bmatrix}^T\begin{bmatrix} 1 & (Ax+b)^T\\(Ax+b) & I\end{bmatrix}\begin{bmatrix}1\\y\end{bmatrix} > 0 \forall y\\
\Leftrightarrow& y^Ty + 2 y^T(Ax+b) + 1 > 0 \;\forall y
\end{align}
$$

With p-satz, the condition $\mathcal{E}\supset\mathcal{S}$ can be imposed as

$$
\begin{align}
-1 - \lambda_0(x, y) g(x) + \lambda_1(x, y) (y^Ty + 2y^T(Ax+b) + 1)\text{ is sos}\\
\lambda_0(x, y), \lambda_1(x, y) \text{ are sos}
\end{align}
$$

where we need to introduce new indeterminates $y$. Hence with the additional indeterminates, the sos constraint in (15)-(16) have significantly larger size than the sos constraint in (1)-(2).
