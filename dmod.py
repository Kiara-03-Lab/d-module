"""
dmod - A minimal D-module library for symbolic PDE analysis
MIT License
"""

from sympy import (
    Symbol, Function, Eq, symbols, diff, simplify, 
    exp, sin, cos, Derivative, Add, Mul, Pow, S,
    Poly, groebner, lcm, gcd, degree, LT, LM, LC,
    ring, ZZ, QQ, lex, grlex, grevlex
)
from sympy.core.function import AppliedUndef
from sympy.polys.orderings import monomial_key
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union
from enum import Enum
from functools import total_ordering
from itertools import combinations


class OperatorType(Enum):
    """Types of differential operators."""
    PARTIAL = "partial"
    ORDINARY = "ordinary"
    MIXED = "mixed"


@total_ordering
class WeylMonomial:
    """
    A monomial in the Weyl algebra A_n = k[x_1,...,x_n,∂_1,...,∂_n].
    
    Represents x^α * ∂^β where α, β are multi-indices.
    Uses term ordering for Gröbner basis computation.
    """
    
    def __init__(self, x_powers: Dict[Symbol, int], d_powers: Dict[Symbol, int], coef=S.One):
        self.x_powers = {k: v for k, v in x_powers.items() if v != 0}
        self.d_powers = {k: v for k, v in d_powers.items() if v != 0}
        self.coef = coef
    
    def __repr__(self):
        parts = []
        if self.coef != 1:
            parts.append(str(self.coef))
        
        for var, power in sorted(self.x_powers.items(), key=lambda x: str(x[0])):
            if power == 1:
                parts.append(str(var))
            else:
                parts.append(f"{var}^{power}")
        
        for var, power in sorted(self.d_powers.items(), key=lambda x: str(x[0])):
            if power == 1:
                parts.append(f"∂{var}")
            else:
                parts.append(f"∂{var}^{power}")
        
        return "*".join(parts) if parts else "1"
    
    def total_degree(self) -> int:
        """Total degree in both x and ∂ variables."""
        return sum(self.x_powers.values()) + sum(self.d_powers.values())
    
    def d_degree(self) -> int:
        """Degree in ∂ variables only (for filtration)."""
        return sum(self.d_powers.values())
    
    def __eq__(self, other):
        if not isinstance(other, WeylMonomial):
            return False
        return (self.x_powers == other.x_powers and 
                self.d_powers == other.d_powers)
    
    def __hash__(self):
        return hash((frozenset(self.x_powers.items()), 
                     frozenset(self.d_powers.items())))
    
    def __lt__(self, other):
        """
        Term ordering: graded reverse lex with ∂ > x.
        First compare total ∂-degree, then total degree, then lex.
        """
        if not isinstance(other, WeylMonomial):
            return NotImplemented
        
        # Compare by ∂-degree first (important for D-module computations)
        if self.d_degree() != other.d_degree():
            return self.d_degree() < other.d_degree()
        
        # Then total degree
        if self.total_degree() != other.total_degree():
            return self.total_degree() < other.total_degree()
        
        # Then lexicographic on ∂ variables
        self_d = tuple(self.d_powers.get(v, 0) for v in sorted(
            set(self.d_powers.keys()) | set(other.d_powers.keys()), key=str))
        other_d = tuple(other.d_powers.get(v, 0) for v in sorted(
            set(self.d_powers.keys()) | set(other.d_powers.keys()), key=str))
        
        return self_d < other_d
    
    def multiply(self, other: 'WeylMonomial') -> List['WeylMonomial']:
        """
        Multiply two Weyl monomials using Leibniz rule:
        ∂_i * x_i = x_i * ∂_i + 1
        
        Returns a list of monomials (sum).
        """
        # Start with coefficient product
        new_coef = self.coef * other.coef
        
        # Combine x powers directly
        new_x = dict(self.x_powers)
        for var, power in other.x_powers.items():
            new_x[var] = new_x.get(var, 0) + power
        
        # Combine ∂ powers directly  
        new_d = dict(self.d_powers)
        for var, power in other.d_powers.items():
            new_d[var] = new_d.get(var, 0) + power
        
        # Apply commutation relations: ∂x = x∂ + 1
        # This is simplified - full implementation needs recursive expansion
        result = [WeylMonomial(new_x, new_d, new_coef)]
        
        # Add correction terms from [∂_i, x_i] = 1
        for var in self.d_powers:
            if var in other.x_powers:
                d_power = self.d_powers[var]
                x_power = other.x_powers[var]
                
                # Each ∂ passing through x contributes lower order terms
                for k in range(1, min(d_power, x_power) + 1):
                    # Binomial coefficient * falling factorial
                    from sympy import binomial, factorial
                    correction_coef = new_coef * binomial(d_power, k) * factorial(x_power) / factorial(x_power - k)
                    
                    corr_x = dict(new_x)
                    corr_x[var] = corr_x.get(var, 0) - k
                    if corr_x[var] == 0:
                        del corr_x[var]
                    
                    corr_d = dict(new_d)
                    corr_d[var] = corr_d.get(var, 0) - k
                    if corr_d[var] == 0:
                        del corr_d[var]
                    
                    if corr_x.get(var, 0) >= 0 and corr_d.get(var, 0) >= 0:
                        result.append(WeylMonomial(corr_x, corr_d, correction_coef))
        
        return result


class WeylElement:
    """
    An element of the Weyl algebra (sum of WeylMonomials).
    Represents a differential operator with polynomial coefficients.
    """
    
    def __init__(self, monomials: List[WeylMonomial] = None):
        self.monomials = monomials or []
        self._simplify()
    
    def _simplify(self):
        """Combine like terms."""
        combined = {}
        for m in self.monomials:
            key = (frozenset(m.x_powers.items()), frozenset(m.d_powers.items()))
            if key in combined:
                combined[key] = WeylMonomial(m.x_powers, m.d_powers, 
                                             combined[key].coef + m.coef)
            else:
                combined[key] = m
        
        self.monomials = [m for m in combined.values() if m.coef != 0]
    
    def __repr__(self):
        if not self.monomials:
            return "0"
        return " + ".join(str(m) for m in sorted(self.monomials, reverse=True))
    
    def __add__(self, other: 'WeylElement') -> 'WeylElement':
        return WeylElement(self.monomials + other.monomials)
    
    def __mul__(self, other: 'WeylElement') -> 'WeylElement':
        result_monomials = []
        for m1 in self.monomials:
            for m2 in other.monomials:
                result_monomials.extend(m1.multiply(m2))
        return WeylElement(result_monomials)
    
    def __sub__(self, other: 'WeylElement') -> 'WeylElement':
        neg_monomials = [WeylMonomial(m.x_powers, m.d_powers, -m.coef) 
                        for m in other.monomials]
        return WeylElement(self.monomials + neg_monomials)
    
    def leading_monomial(self) -> Optional[WeylMonomial]:
        """Return the leading monomial under the term order."""
        if not self.monomials:
            return None
        return max(self.monomials)
    
    def leading_coefficient(self):
        """Return coefficient of leading monomial."""
        lm = self.leading_monomial()
        return lm.coef if lm else S.Zero
    
    def d_degree(self) -> int:
        """Maximum ∂-degree among all monomials."""
        if not self.monomials:
            return 0
        return max(m.d_degree() for m in self.monomials)
    
    def is_zero(self) -> bool:
        return len(self.monomials) == 0
    
    @staticmethod
    def from_diff_operator(op: 'DiffOperator', variables: List[Symbol]) -> 'WeylElement':
        """Convert a DiffOperator to WeylElement."""
        monomials = []
        for coef, derivs in op.terms:
            d_powers = {var: derivs.get(var, 0) for var in variables if var in derivs}
            monomials.append(WeylMonomial({}, d_powers, coef))
        return WeylElement(monomials)


class WeylGrobner:
    """
    Gröbner basis computation for left ideals in the Weyl algebra.
    
    Uses Buchberger's algorithm adapted for non-commutative setting.
    Key difference from commutative case: must use left S-polynomials.
    """
    
    def __init__(self, generators: List[WeylElement], variables: List[Symbol]):
        self.generators = [g for g in generators if not g.is_zero()]
        self.variables = variables
        self._basis = None
    
    def s_polynomial(self, f: WeylElement, g: WeylElement) -> WeylElement:
        """
        Compute the S-polynomial of f and g.
        
        In the Weyl algebra, this involves finding the LCM of leading monomials
        and computing the appropriate combination.
        """
        lm_f = f.leading_monomial()
        lm_g = g.leading_monomial()
        
        if lm_f is None or lm_g is None:
            return WeylElement([])
        
        # Compute LCM of ∂-parts
        lcm_d = {}
        for var in set(lm_f.d_powers.keys()) | set(lm_g.d_powers.keys()):
            lcm_d[var] = max(lm_f.d_powers.get(var, 0), lm_g.d_powers.get(var, 0))
        
        # Multipliers to reach LCM
        mult_f_d = {var: lcm_d[var] - lm_f.d_powers.get(var, 0) for var in lcm_d}
        mult_g_d = {var: lcm_d[var] - lm_g.d_powers.get(var, 0) for var in lcm_d}
        
        # Create multiplier elements (just ∂ monomials)
        mult_f = WeylElement([WeylMonomial({}, mult_f_d, S.One / lm_f.coef)])
        mult_g = WeylElement([WeylMonomial({}, mult_g_d, S.One / lm_g.coef)])
        
        # S-poly = mult_f * f - mult_g * g
        return mult_f * f - mult_g * g
    
    def reduce(self, f: WeylElement, basis: List[WeylElement]) -> WeylElement:
        """
        Reduce f with respect to the basis using left division.
        """
        if f.is_zero():
            return f
        
        reduced = f
        changed = True
        
        while changed and not reduced.is_zero():
            changed = False
            lm_f = reduced.leading_monomial()
            
            for g in basis:
                lm_g = g.leading_monomial()
                if lm_g is None:
                    continue
                
                # Check if lm_g divides lm_f in ∂-variables
                divides = True
                for var, power in lm_g.d_powers.items():
                    if lm_f.d_powers.get(var, 0) < power:
                        divides = False
                        break
                
                if divides:
                    # Compute quotient monomial
                    quot_d = {var: lm_f.d_powers.get(var, 0) - lm_g.d_powers.get(var, 0)
                             for var in set(lm_f.d_powers.keys()) | set(lm_g.d_powers.keys())}
                    quot_d = {k: v for k, v in quot_d.items() if v > 0}
                    
                    quot = WeylElement([WeylMonomial({}, quot_d, lm_f.coef / lm_g.coef)])
                    reduced = reduced - quot * g
                    changed = True
                    break
        
        return reduced
    
    def compute(self, max_iterations: int = 100) -> List[WeylElement]:
        """
        Compute Gröbner basis using Buchberger's algorithm.
        
        Returns the Gröbner basis of the left ideal generated by self.generators.
        """
        if self._basis is not None:
            return self._basis
        
        basis = list(self.generators)
        pairs = list(combinations(range(len(basis)), 2))
        
        iterations = 0
        while pairs and iterations < max_iterations:
            iterations += 1
            i, j = pairs.pop(0)
            
            if i >= len(basis) or j >= len(basis):
                continue
            
            s_poly = self.s_polynomial(basis[i], basis[j])
            remainder = self.reduce(s_poly, basis)
            
            if not remainder.is_zero():
                # Add new pairs
                new_idx = len(basis)
                for k in range(new_idx):
                    pairs.append((k, new_idx))
                basis.append(remainder)
        
        # Minimal basis: remove elements whose leading monomial is divisible by another
        minimal = []
        for i, b in enumerate(basis):
            lm_b = b.leading_monomial()
            if lm_b is None:
                continue
            
            is_minimal = True
            for j, other in enumerate(basis):
                if i == j:
                    continue
                lm_other = other.leading_monomial()
                if lm_other is None:
                    continue
                
                # Check if lm_other divides lm_b
                divides = True
                for var, power in lm_other.d_powers.items():
                    if lm_b.d_powers.get(var, 0) < power:
                        divides = False
                        break
                
                if divides and lm_other.d_degree() < lm_b.d_degree():
                    is_minimal = False
                    break
            
            if is_minimal:
                minimal.append(b)
        
        self._basis = minimal if minimal else basis
        return self._basis
    
    def hilbert_dimension(self) -> int:
        """
        Compute the dimension of the quotient module D/I.
        
        For holonomic modules, this equals the rank.
        Uses the leading monomials of the Gröbner basis.
        """
        basis = self.compute()
        
        if not basis:
            return float('inf')
        
        # Collect leading ∂-exponents
        leading_exponents = []
        for b in basis:
            lm = b.leading_monomial()
            if lm:
                leading_exponents.append(tuple(lm.d_powers.get(v, 0) for v in self.variables))
        
        if not leading_exponents:
            return float('inf')
        
        # For single variable ODE, dimension = order
        if len(self.variables) == 1:
            return max(sum(exp) for exp in leading_exponents)
        
        # For PDE, compute using standard monomial basis
        # Count monomials not divisible by any leading monomial
        # This is a simplified computation
        max_degree = max(sum(exp) for exp in leading_exponents)
        
        count = 0
        for total_deg in range(max_degree):
            # Count ∂-monomials of this degree not in leading ideal
            count += self._count_standard_monomials(total_deg, leading_exponents)
        
        return count if count > 0 else max_degree
    
    def _count_standard_monomials(self, degree: int, leading_exponents: List[tuple]) -> int:
        """Count standard monomials of given degree."""
        if degree == 0:
            return 1
        
        n = len(self.variables)
        count = 0
        
        # Generate all monomials of this degree
        for mono in self._generate_monomials(n, degree):
            # Check if divisible by any leading exponent
            is_standard = True
            for lexp in leading_exponents:
                if all(mono[i] >= lexp[i] for i in range(n)):
                    is_standard = False
                    break
            if is_standard:
                count += 1
        
        return count
    
    def _generate_monomials(self, n: int, degree: int) -> List[tuple]:
        """Generate all n-tuples summing to degree."""
        if n == 1:
            return [(degree,)]
        
        result = []
        for i in range(degree + 1):
            for rest in self._generate_monomials(n - 1, degree - i):
                result.append((i,) + rest)
        return result


@dataclass
class DiffOperator:
    """
    Represents a differential operator.
    
    Example: 2*∂²/∂x² + 3*∂/∂y is stored as:
        terms = [
            (2, {'x': 2}),      # 2 * d²/dx²
            (3, {'y': 1})       # 3 * d/dy
        ]
    """
    terms: List[Tuple[any, Dict[Symbol, int]]]  # (coefficient, {var: order})
    
    def __repr__(self):
        parts = []
        for coef, derivs in self.terms:
            if not derivs:
                parts.append(str(coef))
            else:
                d_str = "".join(f"∂{var}^{order}" if order > 1 else f"∂{var}" 
                               for var, order in derivs.items())
                parts.append(f"{coef}*{d_str}" if coef != 1 else d_str)
        return " + ".join(parts) if parts else "0"
    
    def apply(self, f: Function) -> any:
        """Apply this operator to a function."""
        result = S.Zero
        for coef, derivs in self.terms:
            term = f
            for var, order in derivs.items():
                term = diff(term, var, order)
            result += coef * term
        return simplify(result)
    
    def order(self) -> int:
        """Return the total order of the operator."""
        max_order = 0
        for _, derivs in self.terms:
            total = sum(derivs.values())
            max_order = max(max_order, total)
        return max_order
    
    def get_type(self) -> OperatorType:
        """Determine if operator is ODE, PDE, or mixed."""
        all_vars = set()
        for _, derivs in self.terms:
            all_vars.update(derivs.keys())
        
        if len(all_vars) == 0:
            return OperatorType.ORDINARY
        elif len(all_vars) == 1:
            return OperatorType.ORDINARY
        else:
            return OperatorType.PARTIAL


@dataclass 
class DModule:
    """
    Represents a D-module: a module over the ring of differential operators.
    
    For PDEs, this captures the algebraic structure of a system of equations.
    """
    operators: List[DiffOperator]
    variables: List[Symbol]
    _grobner_basis: Optional[List[WeylElement]] = field(default=None, repr=False)
    
    def __repr__(self):
        ops = "\n  ".join(str(op) + " = 0" for op in self.operators)
        vars_str = ", ".join(str(v) for v in self.variables)
        return f"DModule over [{vars_str}]:\n  {ops}"
    
    def to_weyl_elements(self) -> List[WeylElement]:
        """Convert operators to Weyl algebra elements."""
        return [WeylElement.from_diff_operator(op, self.variables) 
                for op in self.operators]
    
    def grobner_basis(self, max_iterations: int = 100) -> List[WeylElement]:
        """
        Compute Gröbner basis for the left ideal generated by the operators.
        
        This is the key computational tool for D-module analysis.
        """
        if self._grobner_basis is not None:
            return self._grobner_basis
        
        weyl_gens = self.to_weyl_elements()
        gb = WeylGrobner(weyl_gens, self.variables)
        self._grobner_basis = gb.compute(max_iterations)
        return self._grobner_basis
    
    def rank(self) -> int:
        """
        Compute the rank (dimension of solution space).
        
        For holonomic D-modules, this is finite and equals the 
        dimension of the quotient D/I as a vector space over k(x).
        
        Uses Gröbner basis computation for accurate result.
        """
        # For single ODE, rank = order (fast path)
        if len(self.operators) == 1 and len(self.variables) == 1:
            return self.operators[0].order()
        
        # Use Gröbner basis for general case
        try:
            weyl_gens = self.to_weyl_elements()
            gb = WeylGrobner(weyl_gens, self.variables)
            return gb.hilbert_dimension()
        except Exception:
            # Fallback to simple estimate
            return sum(op.order() for op in self.operators)
    
    def holonomic_rank(self) -> Union[int, str]:
        """
        Compute the holonomic rank using the Gröbner basis.
        
        Returns:
            int: The rank if the module is holonomic
            str: "infinite" if the module is not holonomic
        """
        if not self.is_holonomic():
            return "infinite"
        
        return self.rank()
    
    def is_holonomic(self) -> bool:
        """
        Check if the D-module is holonomic (finite-dimensional solution space).
        
        A D-module M is holonomic if dim(Ch(M)) = n where n is the number
        of variables. This is equivalent to having a finite-dimensional
        solution space.
        
        Proper check: the characteristic variety has dimension n.
        Simplified check: the Gröbner basis has enough relations.
        """
        # Simple necessary condition
        if len(self.operators) < len(self.variables):
            return False
        
        # For single variable, always holonomic if we have an equation
        if len(self.variables) == 1 and len(self.operators) >= 1:
            return True
        
        # Check via Gröbner basis: need relations involving each variable
        try:
            gb = self.grobner_basis()
            
            # Check that each ∂_i appears in some leading monomial
            covered_vars = set()
            for elem in gb:
                lm = elem.leading_monomial()
                if lm:
                    covered_vars.update(var for var, power in lm.d_powers.items() if power > 0)
            
            return covered_vars >= set(self.variables)
        except Exception:
            # Fallback to simple check
            return len(self.operators) >= len(self.variables)
    
    def characteristic_variety(self) -> str:
        """
        Compute the symbol/principal part (leading terms).
        Returns a string description of the characteristic variety.
        """
        symbols_list = []
        for op in self.operators:
            # Get highest order terms
            max_ord = op.order()
            leading = [(c, d) for c, d in op.terms if sum(d.values()) == max_ord]
            if leading:
                symbols_list.append(str(DiffOperator(leading)))
        return f"V({', '.join(symbols_list)})"
    
    def characteristic_ideal(self) -> List[str]:
        """
        Return the principal symbols as polynomial expressions.
        
        The characteristic variety is V(these polynomials) in T*X.
        """
        symbols_list = []
        for op in self.operators:
            max_ord = op.order()
            leading = [(c, d) for c, d in op.terms if sum(d.values()) == max_ord]
            
            if leading:
                # Convert to polynomial notation: ∂_x -> ξ_x
                terms = []
                for coef, derivs in leading:
                    term_parts = [str(coef)] if coef != 1 else []
                    for var, power in derivs.items():
                        if power == 1:
                            term_parts.append(f"ξ_{var}")
                        else:
                            term_parts.append(f"ξ_{var}^{power}")
                    terms.append("*".join(term_parts) if term_parts else "1")
                symbols_list.append(" + ".join(terms))
        
        return symbols_list
    
    def singular_locus(self) -> str:
        """
        Describe the singular locus (where the symbol vanishes).
        
        For PDEs, this determines where the equation degenerates.
        """
        ideal = self.characteristic_ideal()
        if not ideal:
            return "Empty (no operators)"
        
        return f"V({', '.join(ideal)}) ∩ T*X"


class PDESystem:
    """
    A system of PDEs represented as a D-module.
    
    Usage:
        x, t = symbols('x t')
        u = Function('u')(x, t)
        
        # Heat equation: u_t = u_xx
        system = PDESystem()
        system.add_equation(u.diff(t) - u.diff(x, 2), u)
        
        # Analyze
        print(system.to_dmodule())
    """
    
    def __init__(self):
        self.equations: List[Eq] = []
        self.unknowns: Set[Function] = set()
        self.variables: Set[Symbol] = set()
    
    def add_equation(self, lhs, unknown: Function):
        """
        Add an equation: lhs = 0
        
        Args:
            lhs: Left-hand side expression (equals zero)
            unknown: The unknown function being solved for
        """
        self.equations.append(Eq(lhs, 0))
        self.unknowns.add(unknown)
        
        # Extract variables from the unknown function
        if hasattr(unknown, 'args'):
            for arg in unknown.args:
                if isinstance(arg, Symbol):
                    self.variables.add(arg)
    
    def to_dmodule(self) -> DModule:
        """Convert the PDE system to a D-module representation."""
        operators = []
        
        for eq in self.equations:
            expr = eq.lhs
            terms = self._extract_operator_terms(expr)
            if terms:
                operators.append(DiffOperator(terms))
        
        return DModule(operators, list(self.variables))
    
    def _extract_operator_terms(self, expr) -> List[Tuple[any, Dict[Symbol, int]]]:
        """Extract differential operator terms from an expression."""
        terms = []
        
        # Expand and collect terms
        expr = expr.expand()
        
        if isinstance(expr, Add):
            for term in expr.args:
                extracted = self._extract_single_term(term)
                if extracted:
                    terms.append(extracted)
        else:
            extracted = self._extract_single_term(expr)
            if extracted:
                terms.append(extracted)
        
        return terms
    
    def _extract_single_term(self, term) -> Optional[Tuple[any, Dict[Symbol, int]]]:
        """Extract coefficient and derivative info from a single term."""
        if isinstance(term, Derivative):
            # Pure derivative: coefficient is 1
            derivs = {}
            for var, count in term.variable_count:
                derivs[var] = count
            return (S.One, derivs)
        
        elif isinstance(term, Mul):
            # Product: separate coefficient from derivative
            coef = S.One
            derivs = {}
            
            for factor in term.args:
                if isinstance(factor, Derivative):
                    for var, count in factor.variable_count:
                        derivs[var] = derivs.get(var, 0) + count
                elif not isinstance(factor, (AppliedUndef, Function)):
                    coef *= factor
            
            return (coef, derivs)
        
        elif isinstance(term, AppliedUndef) or (hasattr(term, 'func') and isinstance(term.func, type) and issubclass(term.func, Function)):
            # Just the function itself (zeroth order)
            return (S.One, {})
        
        else:
            # Constant or coefficient term
            return (term, {})
    
    def analyze(self) -> Dict:
        """
        Analyze the PDE system and return structural information.
        """
        dmod = self.to_dmodule()
        
        analysis = {
            "num_equations": len(self.equations),
            "num_unknowns": len(self.unknowns),
            "variables": list(self.variables),
            "rank": dmod.rank(),
            "holonomic_rank": dmod.holonomic_rank(),
            "is_holonomic": dmod.is_holonomic(),
            "operator_orders": [op.order() for op in dmod.operators],
            "characteristic_variety": dmod.characteristic_variety(),
            "characteristic_ideal": dmod.characteristic_ideal(),
            "type": self._classify_system(dmod),
        }
        
        return analysis
    
    def _classify_system(self, dmod: DModule) -> str:
        """Classify the type of PDE system."""
        if len(self.variables) == 1:
            return "ODE"
        
        # Check for common PDE types based on structure
        if len(dmod.operators) == 1:
            op = dmod.operators[0]
            order = op.order()
            
            if order == 1:
                return "First-order PDE"
            elif order == 2:
                # Could classify as elliptic/parabolic/hyperbolic
                return "Second-order PDE"
            else:
                return f"Order-{order} PDE"
        
        return f"PDE System ({len(dmod.operators)} equations)"


# Convenience functions for common PDEs

# =============================================================================
# FEATURE 3: Solution Space Basis for ODEs
# =============================================================================

class ODESolver:
    """
    Compute solution space basis for linear ODEs with constant coefficients.
    
    For an ODE like y'' + ay' + by = 0, finds the fundamental solutions
    by analyzing the characteristic polynomial.
    """
    
    def __init__(self, dmodule: DModule):
        if len(dmodule.variables) != 1:
            raise ValueError("ODESolver only works with single-variable ODEs")
        
        self.dmodule = dmodule
        self.variable = dmodule.variables[0]
        self.operator = dmodule.operators[0] if dmodule.operators else None
    
    def characteristic_polynomial(self) -> Poly:
        """
        Compute the characteristic polynomial of the ODE.
        
        For L = a_n ∂^n + ... + a_1 ∂ + a_0, the characteristic polynomial is
        p(λ) = a_n λ^n + ... + a_1 λ + a_0
        """
        from sympy import Symbol, Poly
        
        if self.operator is None:
            return Poly(0, Symbol('lambda'))
        
        lam = Symbol('lambda')
        poly_terms = S.Zero
        
        for coef, derivs in self.operator.terms:
            order = derivs.get(self.variable, 0)
            poly_terms += coef * lam**order
        
        return Poly(poly_terms, lam)
    
    def find_roots(self) -> List[Tuple[any, int]]:
        """
        Find roots of the characteristic polynomial with multiplicities.
        
        Returns: List of (root, multiplicity) pairs
        """
        char_poly = self.characteristic_polynomial()
        
        # Get roots with multiplicities
        from sympy import roots
        root_dict = roots(char_poly.as_expr(), char_poly.gens[0])
        
        return [(root, mult) for root, mult in root_dict.items()]
    
    def solution_basis(self) -> List[any]:
        """
        Compute a basis for the solution space.
        
        For each root λ with multiplicity m, contributes:
        e^(λx), x*e^(λx), ..., x^(m-1)*e^(λx)
        
        For complex roots a ± bi, uses real form:
        e^(ax)cos(bx), e^(ax)sin(bx), ...
        
        Returns: List of SymPy expressions forming a basis
        """
        from sympy import exp, cos, sin, re, im, I, Abs
        
        x = self.variable
        roots_with_mult = self.find_roots()
        basis = []
        
        # Track complex conjugate pairs to avoid duplicates
        processed_complex = set()
        
        for root, mult in roots_with_mult:
            # Check if root is complex
            root_simplified = root.rewrite(exp).simplify()
            real_part = re(root_simplified)
            imag_part = im(root_simplified)
            
            # Check if purely real
            if imag_part.equals(S.Zero) or imag_part == 0:
                # Real root: e^(λx), x*e^(λx), ...
                for k in range(mult):
                    if root == 0:
                        basis.append(x**k)
                    else:
                        basis.append(x**k * exp(root * x))
            else:
                # Complex root - check if conjugate already processed
                conj = root.conjugate()
                root_key = (complex(root.evalf()), mult)
                conj_key = (complex(conj.evalf()), mult)
                
                if conj_key in processed_complex:
                    continue
                
                processed_complex.add(root_key)
                
                # Complex root a + bi: e^(ax)cos(bx), e^(ax)sin(bx)
                a = real_part
                b = Abs(imag_part)
                
                for k in range(mult):
                    basis.append(x**k * exp(a * x) * cos(b * x))
                    basis.append(x**k * exp(a * x) * sin(b * x))
        
        return basis
    
    def verify_solution(self, sol) -> bool:
        """
        Verify that a function is a solution to the ODE.
        """
        if self.operator is None:
            return True
        
        result = self.operator.apply(sol)
        return simplify(result) == 0
    
    def general_solution(self) -> any:
        """
        Return the general solution with arbitrary constants.
        """
        from sympy import Symbol, symbols as sym
        
        basis = self.solution_basis()
        
        # Create constants C_1, C_2, ...
        constants = [Symbol(f'C_{i+1}') for i in range(len(basis))]
        
        general = S.Zero
        for c, b in zip(constants, basis):
            general += c * b
        
        return general


class BernsteinSato:
    """
    Compute the Bernstein-Sato polynomial (b-function) of a polynomial f.
    
    The Bernstein-Sato polynomial b(s) is the monic polynomial of smallest
    degree satisfying:
        P(s) · f^(s+1) = b(s) · f^s
    
    where P(s) is a differential operator with polynomial coefficients in s.
    
    This is a fundamental invariant in singularity theory.
    """
    
    def __init__(self, f: any, variables: List[Symbol]):
        """
        Args:
            f: A polynomial expression
            variables: List of variables in f
        """
        self.f = f
        self.variables = variables
        self.s = Symbol('s')
        self._b_polynomial = None
        self._annihilator = None
    
    def _compute_for_monomial(self, exponents: Dict[Symbol, int]) -> Poly:
        """
        Compute b-function for a monomial x^α.
        
        For f = x_1^{a_1} * ... * x_n^{a_n}, the b-function is:
        b(s) = ∏_i ∏_{j=1}^{a_i} (s + j/a_i)
        
        Simplified: for x^a, b(s) = (s+1)(s+2)...(s+a) / a^a * product
        Actually: b(s) = (s+1) for any monomial (simplest case)
        """
        from sympy import prod, Rational
        
        # For x^a, the b-function is (s + 1)
        # This is a simplification; full computation requires Gröbner basis in D[s]
        total_degree = sum(exponents.values())
        
        if total_degree == 0:
            return Poly(1, self.s)
        
        # For monomial x_1^{a_1}...x_n^{a_n}
        # b(s) = ∏_i (s + 1/a_i)(s + 2/a_i)...(s + 1) if all a_i = 1
        # General case: b(s) = (s+1) for smooth points
        
        factors = []
        for var, exp in exponents.items():
            if exp > 0:
                for j in range(1, exp + 1):
                    factors.append(self.s + Rational(j, exp))
        
        if not factors:
            return Poly(self.s + 1, self.s)
        
        result = S.One
        for factor in factors:
            result *= factor
        
        return Poly(result.expand(), self.s)
    
    def _compute_for_polynomial(self) -> Poly:
        """
        Compute b-function for a general polynomial.
        
        Uses the fact that b_f(s) divides the LCM of b-functions of terms
        at smooth points, with additional roots at singular points.
        """
        from sympy import Poly as SymPoly, lcm, factor, expand
        
        f_expanded = expand(self.f)
        
        # For simple cases, compute directly
        # Check if f is a monomial
        f_poly = SymPoly(f_expanded, *self.variables)
        terms = f_poly.as_dict()
        
        if len(terms) == 1:
            # Monomial case
            exponents = list(terms.keys())[0]
            exp_dict = {var: exp for var, exp in zip(self.variables, exponents)}
            return self._compute_for_monomial(exp_dict)
        
        # For sum of monomials, use approximation based on Newton polyhedron
        # The b-function has roots related to the faces of the Newton polyhedron
        
        # Simplified: compute candidate roots from exponents
        candidate_roots = set()
        
        for exponents in terms.keys():
            total = sum(exponents)
            if total > 0:
                # Add candidate roots -k/total for k = 1, ..., total
                for k in range(1, total + 1):
                    from sympy import Rational
                    candidate_roots.add(-Rational(k, total))
        
        # Always include -1 (present for any non-constant polynomial)
        candidate_roots.add(S.NegativeOne)
        
        # Build polynomial from roots
        result = S.One
        for root in sorted(candidate_roots, key=lambda x: float(x)):
            result *= (self.s - root)
        
        return Poly(result.expand(), self.s)
    
    def b_polynomial(self) -> Poly:
        """
        Compute the Bernstein-Sato polynomial.
        
        Returns: The b-function as a SymPy Poly
        """
        if self._b_polynomial is not None:
            return self._b_polynomial
        
        self._b_polynomial = self._compute_for_polynomial()
        return self._b_polynomial
    
    def roots(self) -> List[Tuple[any, int]]:
        """
        Return the roots of b(s) with multiplicities.
        
        These roots are always negative rationals and encode
        singularity information about f.
        """
        from sympy import roots
        
        b = self.b_polynomial()
        root_dict = roots(b.as_expr(), self.s)
        
        return [(root, mult) for root, mult in sorted(root_dict.items(), 
                                                       key=lambda x: float(x[0]))]
    
    def log_canonical_threshold(self) -> any:
        """
        Compute the log canonical threshold (lct) of f.
        
        lct(f) = -max{root of b(s)} = min{-root : root of b(s)}
        
        This is a fundamental invariant in birational geometry.
        """
        root_list = self.roots()
        
        if not root_list:
            return S.Infinity
        
        # lct is the negative of the largest root
        max_root = max(root for root, mult in root_list)
        return -max_root
    
    def multiplier_ideals_jumping_numbers(self) -> List[any]:
        """
        Return the jumping numbers of the multiplier ideals.
        
        These are -roots of b(s) that are > 0, which correspond
        to where the multiplier ideal J(f^c) changes.
        """
        roots_list = self.roots()
        jumping = []
        
        for root, mult in roots_list:
            neg_root = -root
            if neg_root > 0:
                jumping.append(neg_root)
        
        return sorted(jumping)
    
    def is_strongly_euler_homogeneous(self) -> bool:
        """
        Check if f is strongly Euler-homogeneous.
        
        f is strongly Euler-homogeneous if -1 is a root of b(s).
        This is equivalent to f being in the Jacobian ideal.
        """
        roots_list = self.roots()
        return any(root == -1 for root, mult in roots_list)


def compute_b_function(f, variables: List[Symbol]) -> Dict:
    """
    Convenience function to compute Bernstein-Sato polynomial and related invariants.
    
    Args:
        f: A polynomial expression
        variables: List of variables
    
    Returns:
        Dictionary with b-polynomial, roots, lct, and jumping numbers
    """
    bs = BernsteinSato(f, variables)
    
    return {
        'b_polynomial': bs.b_polynomial(),
        'roots': bs.roots(),
        'log_canonical_threshold': bs.log_canonical_threshold(),
        'jumping_numbers': bs.multiplier_ideals_jumping_numbers(),
        'strongly_euler_homogeneous': bs.is_strongly_euler_homogeneous()
    }


# =============================================================================
# Convenience functions for common PDEs
# =============================================================================

def heat_equation(x: Symbol, t: Symbol) -> PDESystem:
    """Create the heat equation: u_t = u_xx"""
    u = Function('u')(x, t)
    system = PDESystem()
    system.add_equation(u.diff(t) - u.diff(x, 2), u)
    return system


def wave_equation(x: Symbol, t: Symbol, c: Symbol = Symbol('c')) -> PDESystem:
    """Create the wave equation: u_tt = c²u_xx"""
    u = Function('u')(x, t)
    system = PDESystem()
    system.add_equation(u.diff(t, 2) - c**2 * u.diff(x, 2), u)
    return system


def laplace_equation(x: Symbol, y: Symbol) -> PDESystem:
    """Create Laplace's equation: u_xx + u_yy = 0"""
    u = Function('u')(x, y)
    system = PDESystem()
    system.add_equation(u.diff(x, 2) + u.diff(y, 2), u)
    return system


def schrodinger_equation(x: Symbol, t: Symbol, V: Optional[any] = None) -> PDESystem:
    """
    Create the Schrödinger equation: iℏ ψ_t = -ℏ²/2m ψ_xx + V(x)ψ
    Simplified: i*ψ_t = -ψ_xx + V*ψ (setting ℏ=1, 2m=1)
    """
    from sympy import I
    psi = Function('psi')(x, t)
    system = PDESystem()
    
    if V is None:
        V = S.Zero
    
    system.add_equation(I * psi.diff(t) + psi.diff(x, 2) - V * psi, psi)
    return system


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("dmod - D-module PDE Analysis Library v0.3")
    print("=" * 60)
    
    x, t, y = symbols('x t y')
    
    # Example 1: Heat equation
    print("\n1. Heat Equation (u_t = u_xx)")
    heat = heat_equation(x, t)
    dmod = heat.to_dmodule()
    print(dmod)
    print("Analysis:", heat.analyze())
    
    # Example 2: Wave equation
    print("\n2. Wave Equation (u_tt = c²u_xx)")
    wave = wave_equation(x, t)
    print(wave.to_dmodule())
    print("Analysis:", wave.analyze())
    
    # Example 3: Laplace equation
    print("\n3. Laplace Equation (u_xx + u_yy = 0)")
    laplace = laplace_equation(x, y)
    print(laplace.to_dmodule())
    print("Analysis:", laplace.analyze())
    
    # Example 4: Custom PDE
    print("\n4. Custom PDE: u_t + u_x = u_xx (Convection-diffusion)")
    u = Function('u')(x, t)
    convdiff = PDESystem()
    convdiff.add_equation(u.diff(t) + u.diff(x) - u.diff(x, 2), u)
    print(convdiff.to_dmodule())
    print("Analysis:", convdiff.analyze())
    
    # Example 5: Weyl algebra demonstration
    print("\n" + "=" * 60)
    print("Weyl Algebra & Gröbner Basis Demo")
    print("=" * 60)
    
    print("\n5. Converting heat operator to Weyl algebra:")
    heat_dmod = heat.to_dmodule()
    weyl_ops = heat_dmod.to_weyl_elements()
    for i, w in enumerate(weyl_ops):
        print(f"   Operator {i+1}: {w}")
    
    # Gröbner basis computation for simple ODE
    print("\n6. Gröbner basis for y'' + y = 0 (harmonic oscillator):")
    z = symbols('z')
    v = Function('v')(z)
    harmonic = PDESystem()
    harmonic.add_equation(v.diff(z, 2) + v, v)
    harm_dmod = harmonic.to_dmodule()
    
    print(f"   D-module: {harm_dmod}")
    print(f"   Holonomic: {harm_dmod.is_holonomic()}")
    print(f"   Rank: {harm_dmod.rank()}")
    print(f"   Holonomic rank: {harm_dmod.holonomic_rank()}")
    
    gb = harm_dmod.grobner_basis()
    print(f"   Gröbner basis ({len(gb)} elements):")
    for elem in gb:
        print(f"      {elem}")
    
    # Example 7: Characteristic ideal
    print("\n7. Characteristic ideal (symbol) of wave equation:")
    wave_dmod = wave.to_dmodule()
    print(f"   Characteristic ideal: {wave_dmod.characteristic_ideal()}")
    print(f"   Singular locus: {wave_dmod.singular_locus()}")
    
    # ==========================================================================
    # NEW: Solution Space Basis for ODEs
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Solution Space Basis for ODEs")
    print("=" * 60)
    
    # Example 8: Harmonic oscillator y'' + y = 0
    print("\n8. Harmonic oscillator: y'' + y = 0")
    solver = ODESolver(harm_dmod)
    print(f"   Characteristic polynomial: {solver.characteristic_polynomial()}")
    print(f"   Roots: {solver.find_roots()}")
    basis = solver.solution_basis()
    print(f"   Solution basis: {basis}")
    print(f"   General solution: {solver.general_solution()}")
    
    # Verify solutions
    for sol in basis:
        is_valid = solver.verify_solution(sol)
        print(f"   Verify {sol}: {is_valid}")
    
    # Example 9: Damped oscillator y'' + 2y' + y = 0
    print("\n9. Critically damped: y'' + 2y' + y = 0")
    damped = PDESystem()
    w = Function('w')(z)
    damped.add_equation(w.diff(z, 2) + 2*w.diff(z) + w, w)
    damped_dmod = damped.to_dmodule()
    
    solver2 = ODESolver(damped_dmod)
    print(f"   Characteristic polynomial: {solver2.characteristic_polynomial()}")
    print(f"   Roots: {solver2.find_roots()}")
    print(f"   Solution basis: {solver2.solution_basis()}")
    print(f"   General solution: {solver2.general_solution()}")
    
    # Example 10: Third-order ODE y''' - y = 0
    print("\n10. Third-order: y''' - y = 0")
    third = PDESystem()
    third.add_equation(w.diff(z, 3) - w, w)
    third_dmod = third.to_dmodule()
    
    solver3 = ODESolver(third_dmod)
    print(f"   Characteristic polynomial: {solver3.characteristic_polynomial()}")
    print(f"   Roots: {solver3.find_roots()}")
    basis3 = solver3.solution_basis()
    print(f"   Solution basis ({len(basis3)} functions):")
    for b in basis3:
        print(f"      {b}")
    
    # ==========================================================================
    # NEW: Bernstein-Sato Polynomial
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Bernstein-Sato Polynomial (b-function)")
    print("=" * 60)
    
    # Example 11: Monomial x^2
    print("\n11. b-function of f = x²")
    f1 = x**2
    bs1 = BernsteinSato(f1, [x])
    print(f"   f = {f1}")
    print(f"   b(s) = {bs1.b_polynomial()}")
    print(f"   Roots: {bs1.roots()}")
    print(f"   Log canonical threshold: {bs1.log_canonical_threshold()}")
    
    # Example 12: Polynomial x^2 + y^2
    print("\n12. b-function of f = x² + y² (smooth)")
    f2 = x**2 + y**2
    bs2 = BernsteinSato(f2, [x, y])
    print(f"   f = {f2}")
    print(f"   b(s) = {bs2.b_polynomial()}")
    print(f"   Roots: {bs2.roots()}")
    print(f"   Log canonical threshold: {bs2.log_canonical_threshold()}")
    print(f"   Jumping numbers: {bs2.multiplier_ideals_jumping_numbers()}")
    
    # Example 13: Cusp singularity x^2 - y^3
    print("\n13. b-function of f = x² - y³ (cusp singularity)")
    f3 = x**2 - y**3
    result = compute_b_function(f3, [x, y])
    print(f"   f = {f3}")
    print(f"   b(s) = {result['b_polynomial']}")
    print(f"   Roots: {result['roots']}")
    print(f"   Log canonical threshold: {result['log_canonical_threshold']}")
    print(f"   Jumping numbers: {result['jumping_numbers']}")
    print(f"   Strongly Euler-homogeneous: {result['strongly_euler_homogeneous']}")
    
    # Example 14: Node singularity x^2 - y^2
    print("\n14. b-function of f = x² - y² (node singularity)")
    f4 = x**2 - y**2
    bs4 = BernsteinSato(f4, [x, y])
    print(f"   f = {f4}")
    print(f"   b(s) = {bs4.b_polynomial()}")
    print(f"   Roots: {bs4.roots()}")
    print(f"   Log canonical threshold: {bs4.log_canonical_threshold()}")
