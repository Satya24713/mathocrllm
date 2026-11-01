# Handwritten Math OCR + Symbolic Studio

A deterministic, sub-second handwriting-to-math workbench combining **`glm-ocr`** (locked in RTX 2050 VRAM) with an extended symbolic ALU engine (**`SymPy`**) and deterministic AST derivation steps in milliseconds.

---

## Supported Mathematical Capabilities

1. **Multivariable Calculus & Coordinate Transformations:**
   - Cartesian triple integrals automatically transformed to spherical coordinates:
     $$\int_0^1 \int_0^{\sqrt{1-x^2}} \int_0^{\sqrt{1-x^2-y^2}} \frac{dx dy dz}{\sqrt{1-x^2-y^2-z^2}} \implies \frac{\pi^2}{8}$$
   - Independent 1D integral products (Euler Beta integrals, rational quartics):
     $$\int_0^\infty \frac{1}{1+x^4} dx \times \int_0^{\pi/2} \sqrt{\cot\theta} d\theta = \frac{\pi}{2\sqrt{2}} \times \frac{\pi}{\sqrt{2}} = \frac{\pi^2}{4}$$
   - Non-greedy differential matching preventing runaway AST variable capture.

2. **Multivariable Jacobians:**
   - Coordinate transformation Jacobians $J\left(\frac{\partial(u, v)}{\partial(x, y)}\right)$, $2 \times 2$, $3 \times 3$, curvilinear systems.

3. **Integral Equations & Parameter Isolation:**
   - Equations containing single, double, and triple integrals:
     $$\int_0^2 \int_0^1 k e^u 2 du dv = 1 \implies k = \frac{1}{4(e - 1)}$$

4. **Systems of Linear & Nonlinear Equations:**
   - Multi-line equations separated by `\\`, `\newline`, commas, or wrapped in `\begin{cases} ... \end{cases}`.
   - Example: $2x + 3y = 0 \\ 4x + 8y = 9 \implies x = -\frac{27}{4}, \quad y = \frac{9}{2}$

5. **Matrices and Linear Algebra:**
   - Detects `\begin{pmatrix}`, `\begin{bmatrix}`, and `\begin{vmatrix}`.
   - Determinants, matrix multiplication, inverses, powers, and transposes.

6. **Limits, Series Summations & Derivatives:**
   - Evaluates limits at points and infinity ($\lim_{x \to 0} \frac{\sin x}{x} = 1$).
   - Finite and infinite series sums ($\sum_{n=1}^{10} n^2 = 385$).
   - Higher order derivatives: $\frac{d^2}{dx^2}(x^4 - 2x^2) = 12x^2 - 4$.

---

## Quickstart

```powershell
python lfm_math_agent.py
```
Open your browser at `http://127.0.0.1:7860`.
