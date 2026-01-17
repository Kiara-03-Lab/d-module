"""Tests for dmod library."""

import pytest
from sympy import symbols, Function, diff, S, exp, cos, sin, sqrt, Rational, I
from dmod import (
    PDESystem, DModule, DiffOperator, WeylMonomial, WeylElement, WeylGrobner,
    ODESolver, BernsteinSato, compute_b_function,
    heat_equation, wave_equation, laplace_equation
)


class TestDiffOperator:
    def test_order(self):
        x = symbols('x')
        # ∂²/∂x²
        op = DiffOperator([(1, {x: 2})])
        assert op.order() == 2
    
    def test_apply(self):
        x = symbols('x')
        f = Function('f')(x)
        
        # ∂/∂x applied to f
        op = DiffOperator([(1, {x: 1})])
        result = op.apply(f)
        assert result == diff(f, x)
    
    def test_mixed_operator(self):
        x, y = symbols('x y')
        # ∂²/∂x∂y
        op = DiffOperator([(1, {x: 1, y: 1})])
        assert op.order() == 2


class TestWeylMonomial:
    def test_creation(self):
        x = symbols('x')
        m = WeylMonomial({x: 2}, {x: 1}, S(3))
        assert m.x_powers[x] == 2
        assert m.d_powers[x] == 1
        assert m.coef == 3
    
    def test_total_degree(self):
        x, y = symbols('x y')
        m = WeylMonomial({x: 2}, {x: 1, y: 1})
        assert m.total_degree() == 4  # 2 + 1 + 1
    
    def test_d_degree(self):
        x, y = symbols('x y')
        m = WeylMonomial({x: 2}, {x: 1, y: 2})
        assert m.d_degree() == 3  # 1 + 2
    
    def test_ordering(self):
        x = symbols('x')
        m1 = WeylMonomial({}, {x: 1})  # ∂x
        m2 = WeylMonomial({}, {x: 2})  # ∂x²
        assert m1 < m2  # higher ∂-degree is larger


class TestWeylElement:
    def test_from_diff_operator(self):
        x = symbols('x')
        op = DiffOperator([(1, {x: 2}), (-1, {})])  # ∂x² - 1
        elem = WeylElement.from_diff_operator(op, [x])
        assert not elem.is_zero()
        assert elem.d_degree() == 2
    
    def test_addition(self):
        x = symbols('x')
        e1 = WeylElement([WeylMonomial({}, {x: 1})])
        e2 = WeylElement([WeylMonomial({}, {x: 2})])
        result = e1 + e2
        assert len(result.monomials) == 2
    
    def test_leading_monomial(self):
        x = symbols('x')
        e = WeylElement([
            WeylMonomial({}, {x: 1}),
            WeylMonomial({}, {x: 2})
        ])
        lm = e.leading_monomial()
        assert lm.d_powers[x] == 2


class TestWeylGrobner:
    def test_simple_ode(self):
        """Test Gröbner basis for y'' + y = 0"""
        x = symbols('x')
        # ∂x² + 1
        gen = WeylElement([
            WeylMonomial({}, {x: 2}),
            WeylMonomial({}, {}, S.One)
        ])
        gb = WeylGrobner([gen], [x])
        basis = gb.compute()
        assert len(basis) >= 1
    
    def test_hilbert_dimension_ode(self):
        """Dimension of D/(∂x² + 1) should be 2"""
        x = symbols('x')
        gen = WeylElement([
            WeylMonomial({}, {x: 2}),
            WeylMonomial({}, {}, S.One)
        ])
        gb = WeylGrobner([gen], [x])
        dim = gb.hilbert_dimension()
        assert dim == 2


class TestODESolver:
    def test_harmonic_oscillator(self):
        """Test y'' + y = 0 has solution basis {cos(x), sin(x)}"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + v, v)
        dmod = system.to_dmodule()
        
        solver = ODESolver(dmod)
        basis = solver.solution_basis()
        
        assert len(basis) == 2
        # Verify solutions
        for sol in basis:
            assert solver.verify_solution(sol)
    
    def test_characteristic_polynomial(self):
        """Test characteristic polynomial computation"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + 2*v.diff(z) + v, v)
        dmod = system.to_dmodule()
        
        solver = ODESolver(dmod)
        char_poly = solver.characteristic_polynomial()
        
        # Should be λ² + 2λ + 1 = (λ + 1)²
        lam = symbols('lambda')
        assert char_poly.as_expr().expand() == (lam**2 + 2*lam + 1)
    
    def test_repeated_roots(self):
        """Test y'' + 2y' + y = 0 (double root at -1)"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + 2*v.diff(z) + v, v)
        dmod = system.to_dmodule()
        
        solver = ODESolver(dmod)
        roots = solver.find_roots()
        
        # Should have root -1 with multiplicity 2
        assert len(roots) == 1
        assert roots[0][0] == -1
        assert roots[0][1] == 2
        
        basis = solver.solution_basis()
        assert len(basis) == 2  # e^(-x) and x*e^(-x)
    
    def test_general_solution(self):
        """Test general solution has correct form"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + v, v)
        dmod = system.to_dmodule()
        
        solver = ODESolver(dmod)
        general = solver.general_solution()
        
        # Should contain C_1 and C_2
        assert symbols('C_1') in general.free_symbols
        assert symbols('C_2') in general.free_symbols


class TestBernsteinSato:
    def test_monomial_x_squared(self):
        """Test b-function of x²"""
        x = symbols('x')
        bs = BernsteinSato(x**2, [x])
        
        roots = bs.roots()
        root_values = [r for r, m in roots]
        
        # Should include -1 and -1/2
        assert S(-1) in root_values
        assert Rational(-1, 2) in root_values
    
    def test_log_canonical_threshold(self):
        """Test lct computation"""
        x = symbols('x')
        bs = BernsteinSato(x**2, [x])
        lct = bs.log_canonical_threshold()
        
        # lct(x²) = 1/2
        assert lct == Rational(1, 2)
    
    def test_strongly_euler_homogeneous(self):
        """Test Euler-homogeneity check"""
        x, y = symbols('x y')
        
        # x² - y³ should be strongly Euler-homogeneous
        bs = BernsteinSato(x**2 - y**3, [x, y])
        assert bs.is_strongly_euler_homogeneous()
    
    def test_jumping_numbers(self):
        """Test jumping numbers computation"""
        x = symbols('x')
        bs = BernsteinSato(x**2, [x])
        
        jumping = bs.multiplier_ideals_jumping_numbers()
        
        # Should include 1/2 and 1
        assert Rational(1, 2) in jumping
        assert S(1) in jumping
    
    def test_compute_b_function_convenience(self):
        """Test convenience function"""
        x, y = symbols('x y')
        result = compute_b_function(x**2 + y**2, [x, y])
        
        assert 'b_polynomial' in result
        assert 'roots' in result
        assert 'log_canonical_threshold' in result
        assert 'jumping_numbers' in result
        assert 'strongly_euler_homogeneous' in result


class TestPDESystem:
    def test_heat_equation_structure(self):
        x, t = symbols('x t')
        heat = heat_equation(x, t)
        
        analysis = heat.analyze()
        assert analysis['num_equations'] == 1
        assert 'characteristic_ideal' in analysis
    
    def test_wave_equation_structure(self):
        x, t = symbols('x t')
        wave = wave_equation(x, t)
        
        analysis = wave.analyze()
        assert analysis['num_equations'] == 1
    
    def test_laplace_equation_structure(self):
        x, y = symbols('x y')
        laplace = laplace_equation(x, y)
        
        analysis = laplace.analyze()
        assert analysis['num_equations'] == 1
    
    def test_custom_pde(self):
        x, t = symbols('x t')
        u = Function('u')(x, t)
        
        system = PDESystem()
        system.add_equation(u.diff(t) - u.diff(x), u)  # Transport equation
        
        dmod = system.to_dmodule()
        assert len(dmod.operators) == 1
        assert dmod.operators[0].order() == 1


class TestDModule:
    def test_to_dmodule_conversion(self):
        x, t = symbols('x t')
        heat = heat_equation(x, t)
        
        dmod = heat.to_dmodule()
        assert isinstance(dmod, DModule)
        assert len(dmod.variables) == 2
    
    def test_characteristic_variety(self):
        x, y = symbols('x y')
        laplace = laplace_equation(x, y)
        
        dmod = laplace.to_dmodule()
        cv = dmod.characteristic_variety()
        assert "V(" in cv
    
    def test_holonomic_ode(self):
        """Single variable ODE should be holonomic"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + v, v)
        
        dmod = system.to_dmodule()
        assert dmod.is_holonomic() == True
        assert dmod.holonomic_rank() == 2
    
    def test_grobner_basis(self):
        """Test Gröbner basis computation"""
        z = symbols('z')
        v = Function('v')(z)
        system = PDESystem()
        system.add_equation(v.diff(z, 2) + v, v)
        
        dmod = system.to_dmodule()
        gb = dmod.grobner_basis()
        assert len(gb) >= 1
    
    def test_to_weyl_elements(self):
        x, t = symbols('x t')
        heat = heat_equation(x, t)
        dmod = heat.to_dmodule()
        
        weyl = dmod.to_weyl_elements()
        assert len(weyl) == 1
        assert not weyl[0].is_zero()
    
    def test_characteristic_ideal(self):
        x, t = symbols('x t')
        wave = wave_equation(x, t)
        dmod = wave.to_dmodule()
        
        ideal = dmod.characteristic_ideal()
        assert len(ideal) >= 1
        assert 'ξ' in ideal[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
