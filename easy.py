"""
dmod.easy - User-friendly interface for D-module computations

No SymPy knowledge required. Just use strings.

Examples:
    >>> from dmod.easy import solve_ode, analyze_pde, b_function
    >>> solve_ode("y'' + y = 0")
    >>> analyze_pde("u_t = u_xx")
    >>> b_function("x^2 - y^3")
"""

import re
from typing import Dict, List, Optional, Union, Any
from sympy import (
    Symbol, Function, symbols, diff, simplify, parse_expr,
    exp, sin, cos, Derivative, S, Poly, sqrt, pi, E, I,
    Rational, latex
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, 
    implicit_multiplication_application, convert_xor
)


# =============================================================================
# String Parsing Engine
# =============================================================================

class EquationParser:
    """
    Parse human-readable differential equations into SymPy expressions.
    
    Supports:
        - Prime notation: y', y'', y'''
        - Subscript notation: u_t, u_xx, u_xy, u_xxy
        - Standard math: +, -, *, /, ^, **
        - Common functions: sin, cos, exp, sqrt
    """
    
    # Standard transformations for parsing
    TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    
    def __init__(self):
        self._var_cache = {}
        self._func_cache = {}
    
    def parse_ode(self, equation: str, func_name: str = 'y', var_name: str = 'x') -> dict:
        """
        Parse an ODE string like "y'' + 2y' + y = 0".
        
        Args:
            equation: ODE as string (e.g., "y'' + y = 0")
            func_name: Name of unknown function (default: 'y')
            var_name: Name of independent variable (default: 'x')
        
        Returns:
            dict with 'lhs', 'func', 'var', 'order'
        """
        # Create symbols
        var = Symbol(var_name)
        func = Function(func_name)(var)
        
        # Split equation at '='
        if '=' in equation:
            lhs_str, rhs_str = equation.split('=', 1)
            lhs_str = lhs_str.strip()
            rhs_str = rhs_str.strip()
        else:
            lhs_str = equation.strip()
            rhs_str = '0'
        
        # Convert prime notation to derivatives
        lhs_expr = self._parse_ode_expr(lhs_str, func, var, func_name)
        rhs_expr = self._parse_ode_expr(rhs_str, func, var, func_name)
        
        # Compute order
        order = self._compute_ode_order(lhs_str, func_name)
        
        return {
            'lhs': lhs_expr - rhs_expr,
            'func': func,
            'var': var,
            'order': order,
            'original': equation
        }
    
    def _parse_ode_expr(self, expr_str: str, func, var, func_name: str):
        """Convert ODE expression string to SymPy expression."""
        s = expr_str
        
        # Replace prime notation: y'''' -> diff(y,x,4), y''' -> diff(y,x,3), etc.
        # Must do longest first
        for n in range(10, 0, -1):
            primes = "'" * n
            pattern = func_name + primes
            if pattern in s:
                s = s.replace(pattern, f"__DERIV_{n}__")
        
        # Replace standalone function name with placeholder
        # Be careful not to replace inside __DERIV__
        s = re.sub(rf'\b{func_name}\b(?!_)', '__FUNC__', s)
        
        # Add implicit multiplication: "2__DERIV" -> "2*__DERIV", "2__FUNC" -> "2*__FUNC"
        s = re.sub(r'(\d)(__DERIV)', r'\1*\2', s)
        s = re.sub(r'(\d)(__FUNC)', r'\1*\2', s)
        
        # Parse the expression
        local_dict = {
            '__FUNC__': func,
            'sin': sin, 'cos': cos, 'exp': exp, 'sqrt': sqrt,
            'pi': pi, 'e': E, 'i': I,
        }
        
        # Add derivative placeholders
        for n in range(1, 11):
            local_dict[f'__DERIV_{n}__'] = diff(func, var, n)
        
        # Add common variable names
        for v in ['x', 'y', 'z', 't', 'a', 'b', 'c', 'k', 'n', 'm']:
            if v != func_name and v not in local_dict:
                local_dict[v] = Symbol(v)
        
        try:
            result = parse_expr(s, local_dict=local_dict, 
                              transformations=self.TRANSFORMATIONS)
            return result
        except Exception as e:
            raise ValueError(f"Could not parse '{expr_str}': {e}")
    
    def _compute_ode_order(self, expr_str: str, func_name: str) -> int:
        """Determine the order of the ODE from the string."""
        max_order = 0
        
        # Count primes
        for match in re.finditer(rf"{func_name}('+)", expr_str):
            order = len(match.group(1))
            max_order = max(max_order, order)
        
        return max_order if max_order > 0 else 1
    
    def parse_pde(self, equation: str, func_name: str = 'u', 
                  var_names: List[str] = None) -> dict:
        """
        Parse a PDE string like "u_t = u_xx" or "u_t + u*u_x = u_xx".
        
        Args:
            equation: PDE as string
            func_name: Name of unknown function (default: 'u')
            var_names: List of variable names (default: auto-detect)
        
        Returns:
            dict with 'lhs', 'func', 'vars', 'order'
        """
        # Auto-detect variables from subscripts
        if var_names is None:
            var_names = self._detect_pde_variables(equation, func_name)
        
        # Create symbols
        vars_syms = [Symbol(v) for v in var_names]
        func = Function(func_name)(*vars_syms)
        
        # Split at '='
        if '=' in equation:
            lhs_str, rhs_str = equation.split('=', 1)
        else:
            lhs_str, rhs_str = equation, '0'
        
        lhs_expr = self._parse_pde_expr(lhs_str.strip(), func, vars_syms, func_name)
        rhs_expr = self._parse_pde_expr(rhs_str.strip(), func, vars_syms, func_name)
        
        order = self._compute_pde_order(equation, func_name)
        
        return {
            'lhs': lhs_expr - rhs_expr,
            'func': func,
            'vars': vars_syms,
            'order': order,
            'original': equation
        }
    
    def _detect_pde_variables(self, equation: str, func_name: str) -> List[str]:
        """Auto-detect variables from subscript notation."""
        # Find all subscripts like u_x, u_tt, u_xy
        pattern = rf'{func_name}_([a-z]+)'
        matches = re.findall(pattern, equation)
        
        # Extract unique variable letters
        vars_set = set()
        for match in matches:
            for char in match:
                vars_set.add(char)
        
        # Default order: x, y, z, t (t last if present)
        result = sorted(vars_set)
        if 't' in result:
            result.remove('t')
            result.append('t')
        
        return result if result else ['x', 't']
    
    def _parse_pde_expr(self, expr_str: str, func, vars_syms: List[Symbol], 
                        func_name: str):
        """Convert PDE expression string to SymPy expression."""
        s = expr_str
        
        # Build variable lookup
        var_dict = {str(v): v for v in vars_syms}
        
        # Find and replace subscript derivatives: u_xxy -> diff(u, x, x, y)
        # Sort by length (longest first) to avoid partial replacements
        deriv_pattern = rf'{func_name}_([a-z]+)'
        derivs = re.findall(deriv_pattern, s)
        derivs_sorted = sorted(set(derivs), key=len, reverse=True)
        
        for subscript in derivs_sorted:
            placeholder = f"__PDE_DERIV_{subscript}__"
            s = s.replace(f"{func_name}_{subscript}", placeholder)
        
        # Replace standalone function
        s = re.sub(rf'\b{func_name}\b(?!_)', '__PDE_FUNC__', s)
        
        # Build local dict
        local_dict = {
            '__PDE_FUNC__': func,
            'sin': sin, 'cos': cos, 'exp': exp, 'sqrt': sqrt,
            'pi': pi, 'e': E, 'i': I,
        }
        local_dict.update(var_dict)
        
        # Add derivative placeholders
        for subscript in derivs_sorted:
            deriv_args = []
            for char in subscript:
                if char in var_dict:
                    deriv_args.append(var_dict[char])
            
            deriv_expr = func
            for v in deriv_args:
                deriv_expr = diff(deriv_expr, v)
            
            local_dict[f"__PDE_DERIV_{subscript}__"] = deriv_expr
        
        try:
            return parse_expr(s, local_dict=local_dict,
                            transformations=self.TRANSFORMATIONS)
        except Exception as e:
            raise ValueError(f"Could not parse '{expr_str}': {e}")
    
    def _compute_pde_order(self, expr_str: str, func_name: str) -> int:
        """Determine the order of the PDE."""
        pattern = rf'{func_name}_([a-z]+)'
        matches = re.findall(pattern, expr_str)
        
        if not matches:
            return 0
        
        return max(len(m) for m in matches)
    
    def parse_polynomial(self, expr_str: str, var_names: List[str] = None) -> dict:
        """
        Parse a polynomial string like "x^2 - y^3".
        
        Args:
            expr_str: Polynomial as string
            var_names: Variable names (default: auto-detect)
        
        Returns:
            dict with 'expr', 'vars'
        """
        # Auto-detect variables
        if var_names is None:
            var_names = self._detect_poly_variables(expr_str)
        
        vars_syms = [Symbol(v) for v in var_names]
        var_dict = {v: sym for v, sym in zip(var_names, vars_syms)}
        
        # Add standard functions and constants
        local_dict = {
            'sin': sin, 'cos': cos, 'exp': exp, 'sqrt': sqrt,
            'pi': pi, 'e': E, 'i': I,
        }
        local_dict.update(var_dict)
        
        try:
            expr = parse_expr(expr_str, local_dict=local_dict,
                            transformations=self.TRANSFORMATIONS)
            return {
                'expr': expr,
                'vars': vars_syms
            }
        except Exception as e:
            raise ValueError(f"Could not parse '{expr_str}': {e}")
    
    def _detect_poly_variables(self, expr_str: str) -> List[str]:
        """Auto-detect variables in polynomial."""
        # Find single letters that aren't function names
        excluded = {'sin', 'cos', 'exp', 'sqrt', 'log', 'tan', 'pi'}
        
        # Find all single letter words
        letters = set(re.findall(r'\b([a-z])\b', expr_str.lower()))
        
        # Filter and sort
        vars_list = sorted(letters - {'e', 'i'})  # exclude e and i (constants)
        
        return vars_list if vars_list else ['x']


# Global parser instance
_parser = EquationParser()


# =============================================================================
# One-Liner Convenience Functions
# =============================================================================

def solve_ode(equation: str, var: str = 'x') -> Dict[str, Any]:
    """
    Solve a linear ODE with constant coefficients.
    
    Args:
        equation: ODE as string (e.g., "y'' + y = 0", "y'' + 2y' + y = 0")
        var: Independent variable name (default: 'x')
    
    Returns:
        dict with keys:
            - 'basis': List of fundamental solutions
            - 'general': General solution with constants C_1, C_2, ...
            - 'order': Order of the ODE
            - 'char_poly': Characteristic polynomial
            - 'roots': Roots with multiplicities
    
    Examples:
        >>> solve_ode("y'' + y = 0")
        {'basis': [cos(x), sin(x)], 'general': 'C_1*cos(x) + C_2*sin(x)', ...}
        
        >>> solve_ode("y'' + 2y' + y = 0")
        {'basis': [exp(-x), x*exp(-x)], ...}
    """
    from dmod import PDESystem, ODESolver
    
    # Parse the equation
    parsed = _parser.parse_ode(equation, var_name=var)
    
    # Create PDESystem
    system = PDESystem()
    system.add_equation(parsed['lhs'], parsed['func'])
    
    # Solve
    dmod = system.to_dmodule()
    solver = ODESolver(dmod)
    
    basis = solver.solution_basis()
    general = solver.general_solution()
    char_poly = solver.characteristic_polynomial()
    roots = solver.find_roots()
    
    return {
        'basis': basis,
        'general': str(general),
        'general_expr': general,
        'order': parsed['order'],
        'char_poly': str(char_poly.as_expr()),
        'char_poly_expr': char_poly,
        'roots': [(str(r), m) for r, m in roots],
        'roots_expr': roots,
        'verified': all(solver.verify_solution(b) for b in basis)
    }


def analyze_pde(equation: str) -> Dict[str, Any]:
    """
    Analyze a PDE and return structural information.
    
    Args:
        equation: PDE as string (e.g., "u_t = u_xx", "u_tt = c^2 * u_xx")
    
    Returns:
        dict with keys:
            - 'type': Classification (parabolic, hyperbolic, elliptic, etc.)
            - 'order': Order of the PDE
            - 'variables': List of variables
            - 'is_holonomic': Whether solution space is finite-dimensional
            - 'characteristic': Characteristic variety description
    
    Examples:
        >>> analyze_pde("u_t = u_xx")
        {'type': 'Second-order PDE (parabolic)', 'order': 2, ...}
        
        >>> analyze_pde("u_xx + u_yy = 0")
        {'type': 'Second-order PDE (elliptic)', ...}
    """
    from dmod import PDESystem
    
    # Parse
    parsed = _parser.parse_pde(equation)
    
    # Create system
    system = PDESystem()
    system.add_equation(parsed['lhs'], parsed['func'])
    
    # Analyze
    analysis = system.analyze()
    dmod = system.to_dmodule()
    
    # Classify type (parabolic/hyperbolic/elliptic for 2nd order)
    pde_type = analysis['type']
    if analysis['operator_orders'] and analysis['operator_orders'][0] == 2:
        pde_type = _classify_second_order_pde(equation, parsed)
    
    return {
        'type': pde_type,
        'order': parsed['order'],
        'variables': [str(v) for v in parsed['vars']],
        'is_holonomic': analysis['is_holonomic'],
        'rank': analysis['rank'],
        'characteristic': analysis['characteristic_variety'],
        'characteristic_ideal': analysis['characteristic_ideal'],
        'equation': equation,
        'dmodule': dmod
    }


def _classify_second_order_pde(equation: str, parsed: dict) -> str:
    """Classify 2nd order PDE as parabolic/hyperbolic/elliptic."""
    # Simple heuristic based on equation structure
    eq_lower = equation.lower().replace(' ', '')
    
    # Heat equation type: u_t = u_xx (parabolic)
    if '_t=' in eq_lower and '_tt' not in eq_lower:
        if '_xx' in eq_lower or '_yy' in eq_lower:
            return "Second-order PDE (parabolic)"
    
    # Wave equation type: u_tt = u_xx (hyperbolic)  
    if '_tt' in eq_lower:
        return "Second-order PDE (hyperbolic)"
    
    # Laplace type: u_xx + u_yy = 0 (elliptic)
    if '_t' not in eq_lower:
        if '_xx' in eq_lower and '_yy' in eq_lower:
            return "Second-order PDE (elliptic)"
    
    return "Second-order PDE"


def b_function(polynomial: str, vars: List[str] = None) -> Dict[str, Any]:
    """
    Compute the Bernstein-Sato polynomial (b-function) of a polynomial.
    
    The b-function encodes singularity information. Its roots are always
    negative rationals.
    
    Args:
        polynomial: Polynomial as string (e.g., "x^2 - y^3", "x^2 + y^2")
        vars: Variable names (default: auto-detect)
    
    Returns:
        dict with keys:
            - 'b_poly': The Bernstein-Sato polynomial b(s)
            - 'roots': Roots with multiplicities
            - 'lct': Log canonical threshold (important singularity invariant)
            - 'jumping': Jumping numbers of multiplier ideals
            - 'singularity_type': Description of singularity type
    
    Examples:
        >>> b_function("x^2")
        {'lct': 0.5, 'roots': [(-1, 1), (-0.5, 1)], ...}
        
        >>> b_function("x^2 - y^3")  # cusp singularity
        {'lct': 1/3, 'singularity_type': 'cusp', ...}
    """
    from dmod import BernsteinSato
    
    # Parse
    parsed = _parser.parse_polynomial(polynomial, vars)
    
    # Compute
    bs = BernsteinSato(parsed['expr'], parsed['vars'])
    
    b_poly = bs.b_polynomial()
    roots = bs.roots()
    lct = bs.log_canonical_threshold()
    jumping = bs.multiplier_ideals_jumping_numbers()
    
    # Classify singularity type
    sing_type = _classify_singularity(polynomial, lct, roots)
    
    return {
        'b_poly': str(b_poly.as_expr()),
        'b_poly_expr': b_poly,
        'roots': [(str(r), m) for r, m in roots],
        'roots_expr': roots,
        'lct': lct,
        'lct_float': float(lct) if lct != S.Infinity else float('inf'),
        'jumping': [str(j) for j in jumping],
        'jumping_expr': jumping,
        'singularity_type': sing_type,
        'is_smooth': lct >= 1,
        'polynomial': polynomial
    }


def _classify_singularity(poly_str: str, lct, roots) -> str:
    """Classify singularity type based on b-function data."""
    poly_lower = poly_str.lower().replace(' ', '').replace('^', '**')
    
    # Check for known types
    if 'x**2-y**3' in poly_lower or 'x^2-y^3' in poly_lower:
        return "cusp (A2)"
    if 'x**2-y**2' in poly_lower or 'x^2-y^2' in poly_lower:
        return "node (A1)"
    if 'x**2+y**2' in poly_lower or 'x^2+y^2' in poly_lower:
        return "smooth point"
    
    # Classify by lct
    if lct >= 1:
        return "smooth or mild singularity"
    elif lct >= Rational(1, 2):
        return "log canonical singularity"
    else:
        return "severe singularity"


# =============================================================================
# ODE Class (Object-Oriented Interface)
# =============================================================================

class ODE:
    """
    User-friendly ODE class.
    
    Examples:
        >>> eq = ODE("y'' + y = 0")
        >>> eq.solve()
        [cos(x), sin(x)]
        >>> eq.general_solution()
        'C_1*cos(x) + C_2*sin(x)'
    """
    
    def __init__(self, equation: str, var: str = 'x'):
        """
        Create an ODE from a string.
        
        Args:
            equation: ODE as string (e.g., "y'' + y = 0")
            var: Independent variable (default: 'x')
        """
        self.equation = equation
        self.var = var
        self._result = None
    
    def _ensure_solved(self):
        if self._result is None:
            self._result = solve_ode(self.equation, self.var)
    
    def solve(self) -> List:
        """Return basis of solutions."""
        self._ensure_solved()
        return self._result['basis']
    
    def general_solution(self) -> str:
        """Return general solution with constants."""
        self._ensure_solved()
        return self._result['general']
    
    def order(self) -> int:
        """Return order of ODE."""
        self._ensure_solved()
        return self._result['order']
    
    def roots(self) -> List:
        """Return roots of characteristic polynomial."""
        self._ensure_solved()
        return self._result['roots']
    
    def char_poly(self) -> str:
        """Return characteristic polynomial."""
        self._ensure_solved()
        return self._result['char_poly']
    
    def explain(self) -> str:
        """Return human-readable explanation."""
        self._ensure_solved()
        r = self._result
        
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  ODE: {self.equation:<52} ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Order: {r['order']:<50} ║",
            f"║  Characteristic polynomial: {r['char_poly']:<28} ║",
            f"║  Roots: {str(r['roots']):<50} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Solution basis:                                         ║",
        ]
        
        for b in r['basis']:
            lines.append(f"║    • {str(b):<52} ║")
        
        lines.extend([
            "╠══════════════════════════════════════════════════════════╣",
            f"║  General solution:                                       ║",
            f"║    {r['general']:<54} ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Verified: {'✓ Yes' if r['verified'] else '✗ No':<47} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ])
        
        return '\n'.join(lines)
    
    def __repr__(self):
        return f"ODE('{self.equation}')"


# =============================================================================
# PDE Class (Object-Oriented Interface)
# =============================================================================

class PDE:
    """
    User-friendly PDE class.
    
    Examples:
        >>> eq = PDE("u_t = u_xx")
        >>> eq.type()
        'Second-order PDE (parabolic)'
        >>> eq.explain()
        # prints formatted analysis
    """
    
    def __init__(self, equation: str):
        """
        Create a PDE from a string.
        
        Args:
            equation: PDE as string (e.g., "u_t = u_xx")
        """
        self.equation = equation
        self._result = None
    
    def _ensure_analyzed(self):
        if self._result is None:
            self._result = analyze_pde(self.equation)
    
    def type(self) -> str:
        """Return PDE classification."""
        self._ensure_analyzed()
        return self._result['type']
    
    def order(self) -> int:
        """Return order of PDE."""
        self._ensure_analyzed()
        return self._result['order']
    
    def variables(self) -> List[str]:
        """Return list of variables."""
        self._ensure_analyzed()
        return self._result['variables']
    
    def is_holonomic(self) -> bool:
        """Check if solution space is finite-dimensional."""
        self._ensure_analyzed()
        return self._result['is_holonomic']
    
    def characteristic(self) -> str:
        """Return characteristic variety."""
        self._ensure_analyzed()
        return self._result['characteristic']
    
    def explain(self) -> str:
        """Return human-readable explanation."""
        self._ensure_analyzed()
        r = self._result
        
        # Explain characteristic ideal symbols
        char_explanation = self._explain_characteristic()
        
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  PDE: {self.equation:<52} ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Type: {r['type']:<51} ║",
            f"║  Order: {r['order']:<50} ║",
            f"║  Variables: {', '.join(r['variables']):<46} ║",
            f"║  Holonomic: {'Yes' if r['is_holonomic'] else 'No':<46} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Characteristic ideal:                                   ║",
        ]
        
        for ideal in r['characteristic_ideal']:
            lines.append(f"║    {ideal:<54} ║")
        
        lines.extend([
            "╠══════════════════════════════════════════════════════════╣",
            "║  Interpretation:                                         ║",
            f"║    {char_explanation:<54} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ])
        
        return '\n'.join(lines)
    
    def _explain_characteristic(self) -> str:
        """Explain what the characteristic variety means."""
        pde_type = self._result['type'].lower()
        
        if 'parabolic' in pde_type:
            return "Diffusion/smoothing behavior (heat-like)"
        elif 'hyperbolic' in pde_type:
            return "Wave propagation with finite speed"
        elif 'elliptic' in pde_type:
            return "Equilibrium/steady-state (boundary-driven)"
        else:
            return "See characteristic ideal for wave behavior"
    
    def __repr__(self):
        return f"PDE('{self.equation}')"


# =============================================================================
# Singularity Class (Object-Oriented Interface)
# =============================================================================

class Singularity:
    """
    Analyze singularities using Bernstein-Sato polynomial.
    
    Examples:
        >>> s = Singularity("x^2 - y^3")
        >>> s.lct()
        1/3
        >>> s.type()
        'cusp (A2)'
    """
    
    def __init__(self, polynomial: str, vars: List[str] = None):
        """
        Create singularity analyzer from polynomial.
        
        Args:
            polynomial: Polynomial as string (e.g., "x^2 - y^3")
            vars: Variable names (default: auto-detect)
        """
        self.polynomial = polynomial
        self.vars = vars
        self._result = None
    
    def _ensure_computed(self):
        if self._result is None:
            self._result = b_function(self.polynomial, self.vars)
    
    def lct(self):
        """Return log canonical threshold."""
        self._ensure_computed()
        return self._result['lct']
    
    def type(self) -> str:
        """Return singularity type."""
        self._ensure_computed()
        return self._result['singularity_type']
    
    def roots(self) -> List:
        """Return roots of b-function."""
        self._ensure_computed()
        return self._result['roots']
    
    def b_poly(self) -> str:
        """Return Bernstein-Sato polynomial."""
        self._ensure_computed()
        return self._result['b_poly']
    
    def jumping_numbers(self) -> List:
        """Return jumping numbers of multiplier ideals."""
        self._ensure_computed()
        return self._result['jumping']
    
    def is_smooth(self) -> bool:
        """Check if point is smooth (lct >= 1)."""
        self._ensure_computed()
        return self._result['is_smooth']
    
    def explain(self) -> str:
        """Return human-readable explanation."""
        self._ensure_computed()
        r = self._result
        
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"║  Polynomial: {self.polynomial:<45} ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Singularity type: {r['singularity_type']:<39} ║",
            f"║  Log canonical threshold: {str(r['lct']):<32} ║",
            f"║  Is smooth: {'Yes' if r['is_smooth'] else 'No':<46} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Bernstein-Sato polynomial b(s):                         ║",
            f"║    {r['b_poly']:<54} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Roots of b(s):                                          ║",
        ]
        
        for root, mult in r['roots']:
            mult_str = f" (multiplicity {mult})" if mult > 1 else ""
            lines.append(f"║    s = {root}{mult_str:<45} ║")
        
        lines.extend([
            "╠══════════════════════════════════════════════════════════╣",
            "║  Jumping numbers (multiplier ideals):                    ║",
            f"║    {', '.join(r['jumping']) if r['jumping'] else 'none':<54} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  Interpretation:                                         ║",
            f"║    {self._interpret():<54} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ])
        
        return '\n'.join(lines)
    
    def _interpret(self) -> str:
        """Provide interpretation of the singularity."""
        lct = self._result['lct']
        
        if self._result['is_smooth']:
            return "Smooth point or very mild singularity"
        elif lct >= Rational(1, 2):
            return "Moderate singularity (log canonical)"
        elif lct >= Rational(1, 3):
            return "Significant singularity"
        else:
            return "Severe singularity"
    
    def __repr__(self):
        return f"Singularity('{self.polynomial}')"


# =============================================================================
# Quick Reference / Help
# =============================================================================

def help_syntax():
    """Print syntax help for equation strings."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                    dmod.easy - Syntax Reference                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ODEs (use prime notation):                                          ║
║    "y'' + y = 0"           →  y'' + y = 0 (harmonic oscillator)     ║
║    "y'' + 2y' + y = 0"     →  y'' + 2y' + y = 0 (damped)           ║
║    "y''' - y = 0"          →  y''' - y = 0 (third order)           ║
║                                                                      ║
║  PDEs (use subscript notation):                                      ║
║    "u_t = u_xx"            →  ∂u/∂t = ∂²u/∂x² (heat equation)      ║
║    "u_tt = c^2 * u_xx"     →  ∂²u/∂t² = c²∂²u/∂x² (wave)          ║
║    "u_xx + u_yy = 0"       →  ∂²u/∂x² + ∂²u/∂y² = 0 (Laplace)     ║
║                                                                      ║
║  Polynomials (for singularity analysis):                             ║
║    "x^2 - y^3"             →  cusp singularity                      ║
║    "x^2 + y^2"             →  smooth point                          ║
║    "x^2 * y + x * y^2"     →  more complex singularity              ║
║                                                                      ║
║  Operators:                                                          ║
║    +, -, *, /              →  standard arithmetic                   ║
║    ^  or  **               →  exponentiation                        ║
║    sin, cos, exp, sqrt     →  standard functions                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # One-liner functions
    'solve_ode',
    'analyze_pde', 
    'b_function',
    
    # Classes
    'ODE',
    'PDE',
    'Singularity',
    
    # Utilities
    'EquationParser',
    'help_syntax',
]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("dmod.easy - User-Friendly Interface Demo")
    print("=" * 60)
    
    # Show syntax help
    help_syntax()
    
    # ODE examples
    print("\n" + "=" * 60)
    print("ODE Examples")
    print("=" * 60)
    
    print("\n1. Harmonic oscillator: y'' + y = 0")
    eq1 = ODE("y'' + y = 0")
    print(eq1.explain())
    
    print("\n2. Damped oscillator: y'' + 2y' + y = 0")
    result2 = solve_ode("y'' + 2y' + y = 0")
    print(f"   Basis: {result2['basis']}")
    print(f"   General: {result2['general']}")
    
    # PDE examples
    print("\n" + "=" * 60)
    print("PDE Examples")
    print("=" * 60)
    
    print("\n3. Heat equation: u_t = u_xx")
    heat = PDE("u_t = u_xx")
    print(heat.explain())
    
    print("\n4. Wave equation: u_tt = u_xx")
    wave = PDE("u_tt = u_xx")
    print(f"   Type: {wave.type()}")
    print(f"   Variables: {wave.variables()}")
    
    print("\n5. Laplace equation: u_xx + u_yy = 0")
    laplace = PDE("u_xx + u_yy = 0")
    print(f"   Type: {laplace.type()}")
    
    # Singularity examples
    print("\n" + "=" * 60)
    print("Singularity Examples")
    print("=" * 60)
    
    print("\n6. Cusp singularity: x^2 - y^3")
    cusp = Singularity("x^2 - y^3")
    print(cusp.explain())
    
    print("\n7. Node singularity: x^2 - y^2")
    node = Singularity("x^2 - y^2")
    print(f"   Type: {node.type()}")
    print(f"   LCT: {node.lct()}")
    
    print("\n8. Quick b-function call:")
    result = b_function("x^2 + y^2")
    print(f"   f = x² + y²")
    print(f"   lct = {result['lct']}")
    print(f"   is_smooth = {result['is_smooth']}")
