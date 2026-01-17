# dmod

A minimal D-module library for symbolic PDE analysis in Python.

## What is this?

`dmod` converts PDEs into algebraic objects (D-modules), letting you analyze differential equations structurally without solving them explicitly.

**v0.4:** Now includes `dmod.easy` — no SymPy knowledge required!

## Quick Start (Easy API)

```python
from easy import solve_ode, analyze_pde, b_function

# Solve ODE with one line
result = solve_ode("y'' + y = 0")
print(result['basis'])      # [cos(x), sin(x)]
print(result['general'])    # C_1*cos(x) + C_2*sin(x)

# Analyze PDE with one line
info = analyze_pde("u_t = u_xx")
print(info['type'])         # Second-order PDE (parabolic)

# Compute b-function with one line
b = b_function("x^2 - y^3")
print(b['lct'])             # 1/3 (log canonical threshold)
```

## Object-Oriented Interface

```python
from easy import ODE, PDE, Singularity

# ODE class
eq = ODE("y'' + 2y' + y = 0")
eq.solve()              # [exp(-x), x*exp(-x)]
eq.general_solution()   # C_1*exp(-x) + C_2*x*exp(-x)
eq.explain()            # Pretty-printed analysis

# PDE class
heat = PDE("u_t = u_xx")
heat.type()             # Second-order PDE (parabolic)
heat.variables()        # ['x', 't']
heat.explain()          # Pretty-printed analysis

# Singularity class
cusp = Singularity("x^2 - y^3")
cusp.lct()              # 1/3
cusp.type()             # cusp (A2)
cusp.explain()          # Pretty-printed analysis
```

## Syntax Reference

| Type | Syntax | Example |
|------|--------|---------|
| ODE | Prime notation | `y'' + 2y' + y = 0` |
| PDE | Subscript notation | `u_t = u_xx`, `u_xx + u_yy = 0` |
| Polynomial | Standard math | `x^2 - y^3`, `x^2 + y^2` |

Supported operators: `+`, `-`, `*`, `/`, `^` (or `**`), `sin`, `cos`, `exp`, `sqrt`

## Install

```bash
pip install sympy
# Then copy dmod.py and easy.py to your project
```

---

## Advanced API (Full SymPy)

For power users who need direct access to D-module structures:

```python
from sympy import symbols, Function
from dmod import PDESystem, ODESolver, BernsteinSato

# Create PDE system manually
x, t = symbols('x t')
u = Function('u')(x, t)
system = PDESystem()
system.add_equation(u.diff(t) - u.diff(x, 2), u)

# Get D-module
dmod = system.to_dmodule()

# Compute Gröbner basis
gb = dmod.grobner_basis()

# Get Weyl algebra representation
weyl = dmod.to_weyl_elements()
```

## Full API Reference

### Easy Module (`easy.py`)

**One-liners:**
- `solve_ode(equation)` → dict with basis, general solution, roots
- `analyze_pde(equation)` → dict with type, order, variables
- `b_function(polynomial)` → dict with lct, roots, singularity type

**Classes:**
- `ODE(equation)` → `.solve()`, `.general_solution()`, `.explain()`
- `PDE(equation)` → `.type()`, `.variables()`, `.explain()`
- `Singularity(polynomial)` → `.lct()`, `.type()`, `.explain()`

### Core Module (`dmod.py`)

**PDESystem:**
- `add_equation(lhs, unknown)` — Add equation
- `to_dmodule()` — Convert to D-module
- `analyze()` — Get analysis dict

**DModule:**
- `rank()` — Solution space dimension
- `is_holonomic()` — Finite-dimensional?
- `grobner_basis()` — Compute Gröbner basis
- `characteristic_ideal()` — Principal symbols

**ODESolver:**
- `characteristic_polynomial()` — Char poly
- `solution_basis()` — Fundamental solutions
- `general_solution()` — With constants

**BernsteinSato:**
- `b_polynomial()` — The b-function
- `roots()` — Roots with multiplicities
- `log_canonical_threshold()` — LCT

## Roadmap

- [x] Gröbner bases for D-modules
- [x] Holonomic rank computation
- [x] Solution space basis (ODEs)
- [x] Bernstein-Sato polynomial
- [x] **Easy string-based API**
- [x] **One-liner convenience functions**
- [x] **Human-readable explain() output**
- [ ] Variable coefficient ODEs
- [ ] Plotting support

## License

MIT
