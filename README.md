# Compatible CLF and CBFs
Verifying and synthesizing compatible Control Lyapunov Function (CLF) and Control Barrier Function (CBF) through Sum-of-Squares.

# Background
For a continuous-time control affine system
$$\dot{x} = f(x) + g(x)u, u\in\mathcal{U},$$
we want to synthesize a controller that can stabilize the system, while also maintain safety. To this end, we will introduce Control Lyapunov Function (CLF) for stability, and Control Barrier Function (CBF) for safety.

The Control Lyapunov Function (CLF) $V(x)$ satisfies

$$ \exists u\in\mathcal{U}, L_f V(x) + L_g V(x)u \leq -\kappa_VV(x)$$

where $L_fV(x)$ is the Lie-derivative $\frac{\partial V}{\partial x}f(x)$, similarly $L_gV(x)$ is the Lie-derivative $\frac{\partial V}{\partial x}g(x)$. $\kappa_V>0$ is a given constant.

The Control Barrier Function (CBF) $b(x)$ satisfies

$$ \exists u \in\mathcal{U}, L_fb(x) + L_gb(x)u \geq -\kappa_bb(x),$$

$\kappa_b>0$ is a given constant.

## Compatible CLF/CBF
We say a CLF $V(x)$ is compatible with a CBF $b(x)$ if there exists a common action $u$ that satisfies both the CLF condition and the CBF condition simultaneously

$$\exists u\in\mathcal{U} \begin{cases}L_fV(x)+L_gV(x)u\leq -\kappa_VV(x)\\
 L_fb(x)+L_gb(x)u\geq-\kappa_bb(x)\end{cases}.$$

We will certify and synthesize such compatible CLF and CBFs through Sum-of-Squares optimization.

# Getting started
After installing all the dependencies in `requirements.txt`, please make sure that you can use Mosek solver. Please set the environment variable that points to the Mosek license as
```
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

## Using Drake
We formulate and solve the optimization problem using [Drake](https://drake.mit.edu). To get started with Drake, you can checkout its tutorials (on the Drake webpage, navigate to "Resources"->"Tutorials"). Drake hosts several tutorials on deepnote, and there is a section of tutorials on Mathematical Programming. You might want to pay special attention to "Sum-of-squares optimization" as we will use it in this project.

# Contributing
To maintain quality of the code, we suggest in each pull request, please do the followings
- Add unit tests. We use [pytest](https://docs.pytest.org/en) framework.
- Format the code. You can run 
  ```
  black ./
  ```
  to format each python code using `black` formatter.

Each pull request need to pass the CI before being merged. We also use [reviewable](reviewable.io) to review the code.
