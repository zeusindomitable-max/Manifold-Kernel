# Mathematical Foundation of Heat-Kernel Regularization

---

## 1. Core Theory

### 1.1 Heat Kernel on Riemannian Manifolds
Let $(\mathcal{M}, g)$ be an $n$-dimensional Riemannian manifold with local coordinates $\theta^i$ and metric tensor $g_{ij}(\theta)$.  
The **heat kernel** $K_\tau(\theta, \theta')$ is the fundamental solution of the heat equation:

\[
\partial_\tau K_\tau = \Delta_g K_\tau, \quad K_0(\theta, \theta') = \delta_g(\theta, \theta')
\]

where $\Delta_g$ is the **Laplace–Beltrami operator**:

\[
\Delta_g f = |g|^{-1/2}\partial_i \left(|g|^{1/2} g^{ij} \partial_j f\right)
\]

and $\delta_g$ is the **Dirac delta** distribution compatible with the manifold volume form  
$dV_g = \sqrt{|g|}\,d^n\theta$.

---

### 1.2 Seeley–DeWitt Asymptotic Expansion
For small diffusion time $\tau \to 0^+$, the heat kernel admits an asymptotic expansion of the form:

\[
K_\tau(\theta, \theta') \sim (4\pi\tau)^{-n/2} e^{-d_g^2(\theta, \theta') / 4\tau}
\sum_{k=0}^{\infty} a_k(\theta, \theta') \tau^k
\]

where:
- $d_g(\theta, \theta')$ is the geodesic distance on $\mathcal{M}$,
- $a_k$ are the **Seeley–DeWitt coefficients**, geometric invariants depending on curvature.

At coincidence limit $\theta \to \theta'$,
the first few coefficients encode intrinsic curvature information of $\mathcal{M}$.

---

## 2. Regularized Curvature Field

### 2.1 Definition
The **heat-kernel regularized curvature field** is defined as a smoothed curvature observable:

\[
\Phi_\tau^{\text{reg}}(R)(\theta) = 
\int_\mathcal{M} K_\tau(\theta, \theta') R(\theta') \, dV_g(\theta') 
- \text{counterterms}
\]

where $R(\theta')$ is the scalar curvature field.

---

### 2.2 Counterterm Subtraction
To eliminate ultraviolet divergences as $\tau \to 0^+$, we subtract the first few terms of the asymptotic expansion:

\[
\text{counterterms} = (4\pi\tau)^{-n/2} 
\sum_{k=0}^3 a_k \, \overline{R}_k \, \tau^k
- c \log(\mu^2 \tau) \int \operatorname{tr}(a_2) \, dV_g
\]

where $\mu$ is the renormalization scale and $c$ a scheme-dependent constant.

---

### 2.3 Seeley–DeWitt Coefficients (for $n=2$)
For two-dimensional manifolds such as $S^2$:
\[
\begin{aligned}
a_0 &= 1, \\
a_1 &= E + \frac{1}{6} R, \\
a_2 &= \frac{1}{2} E^2 + \frac{1}{6} \Box E + \frac{1}{72} R^2 + \cdots
\end{aligned}
\]
where $E$ encodes potential-like curvature corrections.

---

## 3. Variational Formulation

The **heat-kernel regularization** can be embedded in a geometric variational principle:

\[
\mathcal{S}[\theta, \tau] = 
\int_{t_0}^{t_1} \int_\Theta
\left[
\mathcal{L}_{\text{task}}(\theta)
+ \alpha \| \mathrm{Ric}(\theta) \|^2
+ \beta \, \Phi_\tau^{\text{reg}}(R)(\theta)
+ \frac{\lambda}{2} (\partial_t \tau)^2
\right]
dV_g \, dt
\]

---

### 3.1 Euler–Lagrange Equations
Variation with respect to $\theta_p$ gives:

\[
\frac{\delta\mathcal{S}}{\delta\theta_p} =
\int_\Theta
\left[
\frac{\partial L}{\partial\theta_p}
- \nabla_i\!\left(
\frac{\partial L}{\partial(\nabla_i \theta_p)}
\right)
\right]
dV_g
+ \frac{1}{2} \int L \, g^{ij} \frac{\partial g_{ij}}{\partial \theta_p} \, dV_g
+ \int
\frac{\delta K_\tau}{\delta g_{ij}}
\frac{\partial g_{ij}}{\partial \theta_p}
\frac{\partial L}{\partial K_\tau}
\, dV_g
\]

Variation with respect to $\tau$ yields the **diffusion-scale evolution**:

\[
\lambda \, \ddot{\tau} = -\beta \, \partial_\tau \Phi_\tau^{\text{reg}}
\]

and in the overdamped (gradient-flow) limit:

\[
\tau \leftarrow \tau - \eta_\tau \beta \, \partial_\tau \Phi_\tau^{\text{reg}} + \xi(t)
\]
where $\xi(t)$ represents stochastic thermal noise or Monte Carlo diffusion.

---

## 4. Numerical Implementation

### 4.1 Gaussian Kernel Approximation
In local normal coordinates, the heat kernel can be approximated as:

\[
K_\tau(\theta, \theta') \approx 
(4\pi\tau)^{-n/2} \exp\!\left(
-\frac{(\theta - \theta')^\top F(\bar{\theta}) P_k (\theta - \theta')}{4\tau}
\right)
\]

where:
- $F(\bar{\theta})$ is the **Fisher information metric** (or empirical covariance),
- $P_k$ projects onto the dominant $k$ eigenmodes of the local metric tensor.

This approximation allows **local geodesic heat diffusion** to be simulated using tensor operations in PyTorch.

---

### 4.2 Hutchinson’s Stochastic Trace Estimation
For efficient curvature and Hessian trace computations, the **Hutchinson estimator** is applied:

\[
\widehat{\operatorname{Tr}}(H)
= \frac{1}{m} \sum_{i=1}^m v_i^\top H v_i,
\quad v_i \sim \{\pm 1\}^P
\]

This estimator is unbiased and parallelizable, enabling large-scale geometric regularization without explicitly forming the full Hessian matrix.

---

## 5. Connection to the Codebase

- **`SphereManifold.metric()`** implements the local metric tensor \( g_{ij} = \text{diag}(r^2, r^2 \sin^2\theta) \).
- **`HeatKernelRegularizer.forward()`** corresponds to numerical evaluation of the regularized curvature field \( \Phi_\tau^{\text{reg}} \).
- **Seeley–DeWitt coefficients** influence the choice of regularization constants (`α`, `β`, `λ`) in the training or optimization routines.

---

## 6. References
1. DeWitt, B.S. *Dynamical Theory of Groups and Fields*, Gordon and Breach (1965).  
2. Vassilevich, D.V. *Heat Kernel Expansion: User’s Manual*, *Physics Reports* **388**, 279–360 (2003).  
3. Rosenberg, S. *The Laplacian on a Riemannian Manifold*, Cambridge University Press (1997).  
4. Gilkey, P. *Invariance Theory, the Heat Equation, and the Atiyah–Singer Index Theorem*, CRC Press (1995).  
5. Coifman & Lafon, *Diffusion Maps*, *Applied and Computational Harmonic Analysis* (2006).  
6. Smola et al., *Metrics for Deep Manifold Learning via Heat Kernels*, NeurIPS (2022).
