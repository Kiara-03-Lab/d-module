"""Tests for dmod.easy module."""

import pytest
from sympy import cos, sin, exp, Symbol, Rational
from easy import (
    solve_ode, analyze_pde, b_function,
    ODE, PDE, Singularity,
    EquationParser
)


class TestEquationParser:
    def test_parse_simple_ode(self):
        parser = EquationParser()
        result = parser.parse_ode("y'' + y = 0")
        assert result['order'] == 2
        assert result['var'] == Symbol('x')
    
    def test_parse_ode_with_coefficients(self):
        parser = EquationParser()
        result = parser.parse_ode("y'' + 2y' + y = 0")
        assert result['order'] == 2
    
    def test_parse_pde_heat(self):
        parser = EquationParser()
        result = parser.parse_pde("u_t = u_xx")
        assert result['order'] == 2
        assert len(result['vars']) == 2
    
    def test_parse_pde_laplace(self):
        parser = EquationParser()
        result = parser.parse_pde("u_xx + u_yy = 0")
        assert result['order'] == 2
    
    def test_parse_polynomial(self):
        parser = EquationParser()
        result = parser.parse_polynomial("x^2 - y^3")
        assert len(result['vars']) == 2


class TestSolveODE:
    def test_harmonic_oscillator(self):
        result = solve_ode("y'' + y = 0")
        assert result['order'] == 2
        assert len(result['basis']) == 2
        assert result['verified'] == True
    
    def test_damped_oscillator(self):
        result = solve_ode("y'' + 2y' + y = 0")
        assert result['order'] == 2
        assert len(result['basis']) == 2
        # Check for exp(-x) in basis
        basis_str = str(result['basis'])
        assert 'exp' in basis_str
    
    def test_first_order(self):
        result = solve_ode("y' - y = 0")
        assert result['order'] == 1
        assert len(result['basis']) == 1
    
    def test_third_order(self):
        result = solve_ode("y''' - y = 0")
        assert result['order'] == 3
        assert len(result['basis']) == 3


class TestAnalyzePDE:
    def test_heat_equation(self):
        result = analyze_pde("u_t = u_xx")
        assert result['order'] == 2
        assert 'parabolic' in result['type'].lower()
        assert 'x' in result['variables']
        assert 't' in result['variables']
    
    def test_wave_equation(self):
        result = analyze_pde("u_tt = u_xx")
        assert result['order'] == 2
        assert 'hyperbolic' in result['type'].lower()
    
    def test_laplace_equation(self):
        result = analyze_pde("u_xx + u_yy = 0")
        assert result['order'] == 2
        assert 'elliptic' in result['type'].lower()


class TestBFunction:
    def test_monomial(self):
        result = b_function("x^2")
        assert result['lct'] == Rational(1, 2)
    
    def test_cusp(self):
        result = b_function("x^2 - y^3")
        assert result['lct'] == Rational(1, 3)
        assert result['singularity_type'] == "cusp (A2)"
    
    def test_node(self):
        result = b_function("x^2 - y^2")
        assert result['lct'] == Rational(1, 2)
        assert result['singularity_type'] == "node (A1)"
    
    def test_smooth(self):
        result = b_function("x^2 + y^2")
        assert result['lct'] == Rational(1, 2)


class TestODEClass:
    def test_solve(self):
        eq = ODE("y'' + y = 0")
        basis = eq.solve()
        assert len(basis) == 2
    
    def test_general_solution(self):
        eq = ODE("y'' + y = 0")
        gen = eq.general_solution()
        assert 'C_1' in gen
        assert 'C_2' in gen
    
    def test_order(self):
        eq = ODE("y''' - y = 0")
        assert eq.order() == 3
    
    def test_explain(self):
        eq = ODE("y'' + y = 0")
        explanation = eq.explain()
        assert 'harmonic' in explanation.lower() or 'cos' in explanation


class TestPDEClass:
    def test_type(self):
        eq = PDE("u_t = u_xx")
        assert 'parabolic' in eq.type().lower()
    
    def test_variables(self):
        eq = PDE("u_t = u_xx")
        vars = eq.variables()
        assert 'x' in vars
        assert 't' in vars
    
    def test_explain(self):
        eq = PDE("u_t = u_xx")
        explanation = eq.explain()
        assert 'PDE' in explanation


class TestSingularityClass:
    def test_lct(self):
        s = Singularity("x^2 - y^3")
        assert s.lct() == Rational(1, 3)
    
    def test_type(self):
        s = Singularity("x^2 - y^3")
        assert s.type() == "cusp (A2)"
    
    def test_is_smooth(self):
        s = Singularity("x^2 - y^3")
        assert s.is_smooth() == False
    
    def test_explain(self):
        s = Singularity("x^2 - y^3")
        explanation = s.explain()
        assert 'cusp' in explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
