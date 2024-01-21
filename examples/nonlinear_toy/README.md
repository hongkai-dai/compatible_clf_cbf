We search for the compatible CLF and CBF function of a nonlinear dynamical system with 2 states

$$
\begin{align*}
\dot{x}_0 =& u\\
\dot{x}_1 =& -x_0 + \frac{1}{6}x_0^3 - u
\end{align*}
$$
We will consider the case with or without the input limits on $u$.

This system is introduced in *Searching for control Lyapunov functions using sums-of-squares programming* by Weehong Tan and Andrew Packard.

I _think_ this system is the taylor expansion of the following system
$$
\begin{align*}
\dot{x}_0 = &u\\
\dot{x}_1=&-\sin x_0 -u
\end{align*}
$$

So another way to model this system is to use the trigonometric state
$$
\begin{align*}
\bar{x}_0 =& \sin x_0\\
\bar{x}_1 =& \cos x_0 -1\\
\bar{x}_2 =& x_1
\end{align*}
$$ 
Note that the new state $\bar{x}$ has the equilibrium $\bar{x}^*=0$.
The dynamics is
$$
\begin{align*}
\dot{\bar{x}}_0 =& (\bar{x}_1+1)u\\
\dot{\bar{x}}_1=&-\bar{x}_0u\\
\dot{\bar{x}}_2=&-\bar{x}_0 -u
\end{align*}
$$
