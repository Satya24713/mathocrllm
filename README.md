# Handwritten Math OCR + Symbolic Studio

A deterministic, sub-second handwriting-to-math workbench combining **`glm-ocr`** (locked in RTX 2050 VRAM) with an extended symbolic ALU engine (**`SymPy`**), and optional **`lfm3`** derivation synthesis.

---

## Supported Mathematical Capabilities

1. **Systems of Linear & Nonlinear Equations:**
   - Multi-line equations separated by `\\`, `\newline`, commas, or wrapped in `\begin{cases} ... \end{cases}`.
   - Example: $2x + 3y = 0 \\ 4x + 8y = 9 \implies x = -\frac{27}{4}, \quad y = \frac{9}{2}$
   - Handles 2-variable, 3-variable, and $n$-variable systems.

2. **Matrices and Linear Algebra:**
   - Detects `\begin{pmatrix}`, `\begin{bmatrix}`, and `\begin{vmatrix}`.
   - **Determinants:** `\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix} \implies -2`
   - **Matrix Inverses:** `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1}`
   - Dimensions, trace, and symbolic matrix representations.

3. **Limits:**
   - Evaluates limits at points and infinity ($\lim_{x \to a}$, $\lim_{x \to \infty}$, $\lim_{x \to 0}$).
   - Example: $\lim_{x \to 0} \frac{\sin x}{x} = 1$

4. **Series Summations & Products:**
   - Evaluates finite sums, infinite closed-form series, and products ($\sum_{n=1}^k$, $\sum_{n=1}^\infty$, $\prod_{n=1}^k$).
   - Example: $\sum_{n=1}^{10} n^2 = 385$

5. **Higher-Order & Partial Derivatives:**
   - Evaluates $n$-th order derivatives: $\frac{d^2}{dx^2}(x^4 - 2x^2) = 12x^2 - 4$.

6. **Partial Fraction Decomposition:**
   - Computes partial fractions via `apart`: $\frac{1}{x^2 - 1} = -\frac{1}{2(x+1)} + \frac{1}{2(x-1)}$.

7. **Trigonometric & Hyperbolic Identities:**
   - Simplification of trigonometric and hyperbolic expressions: $\sin^2 x + \cos^2 x = 1$.

8. **Polynomial Expansion & Factoring:**
   - Factoring: $x^3 - 8 = (x - 2)(x^2 + 2x + 4)$.
   - Expansion: $(x + 2)^3 = x^3 + 6x^2 + 12x + 8$.

9. **Single Equations & Calculus:**
   - Algebraic, quadratic, cubic, transcendental equations.
   - Definite and indefinite integrals ($\int_0^3 x^2 dx = 9$).

---

## Quickstart

```powershell
python lfm_math_agent.py
```
Open your browser at `http://127.0.0.1:7860`.
