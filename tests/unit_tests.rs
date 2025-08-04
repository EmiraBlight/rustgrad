#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use rustgrad::Node;
    #[test]
    fn values_are_created() {
        let a = Node::new(4.0);
        assert_eq!(a.data(), 4.0);
    }
    #[test]
    fn values_are_adding() {
        let a = Node::new(4.0);
        let b = Node::new(3.0);
        let c = a + b;
        assert_eq!(c.data(), 7.0);
    }
    #[test]
    fn values_are_subtracting() {
        let a = Node::new(5.0);
        let b = Node::new(2.0);
        let c = a - b;
        assert_eq!(c.data(), 3.0);
    }
    #[test]
    fn values_are_multiplying() {
        let a = Node::new(5.0);
        let b = Node::new(3.0);
        let c = a * b;
        assert_eq!(c.data(), 15.0);
    }
    #[test]
    fn values_derive_mult() {
        let a = Node::new(4.0);
        let b = Node::new(-2.0);
        let c = a.clone() * b.clone();
        c.backward();
        assert_eq!(a.grad(), -2.0);
        assert_eq!(b.grad(), 4.0);
        assert_eq!(c.data(), -8.0);
    }
    #[test]
    fn values_derive_adding() {
        let a = Node::new(12.0);
        let b = Node::new(10.0);
        let c = a.clone() + b.clone();
        //these are not cloned in pratice. Cloned for testing purpose
        assert_eq!(c.data(), 22.0);

        c.backward();

        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
    }
    #[test]
    fn test_from_video() {
        let a = Node::new(2.0);
        let b = Node::new(-3.0);
        let c = Node::new(10.0);
        let e = a.clone() * b.clone();
        let d = e.clone() + c.clone();
        let f = Node::new(-2.0);
        let l = d.clone() * f.clone();
        l.backward();
        assert_eq!(l.data(), -8.0);
        assert_eq!(d.grad(), -2.0);
        assert_eq!(f.grad(), 4.0);
        assert_eq!(c.grad(), -2.0);
        assert_eq!(e.grad(), -2.0);
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), -4.0);
    }
    #[test]
    fn nodes_are_diving() {
        let a = Node::new(12.0);
        let b = Node::new(3.0);
        let c = a / b;
        assert_eq!(c.data(), 4.0);
    }
    #[test]
    fn nodes_are_diving_and_deriving() {
        let a = Node::new(12.0);
        let b = Node::new(3.0);
        let c = a.clone() / b.clone();
        c.backward();
        assert_eq!(c.data(), 4.0);
        assert_approx_eq!(a.grad(), 0.333333, 0.1);
        assert_approx_eq!(b.grad(), -1.333333, 0.1);
    }
    #[test]
    fn nodes_can_negate() {
        let a = Node::new(3.0);
        let b = -a;
        assert_eq!(b.data(), -3.0);
    }

    #[test]
    fn test_pow_forward() {
        let a = Node::new(2.0);
        let b = Node::new(3.0);
        let c = a.pow(b);
        assert_approx_eq!(c.data(), 8.0, 1e-6);
    }

    #[test]
    fn test_pow_backward_constant_exponent() {
        let x = Node::new(2.0);
        let exponent = Node::new(3.0);
        let y = x.pow(exponent.clone());
        y.backward();

        // dy/dx = n * x^(n-1) = 3 * 2^2 = 12
        assert_approx_eq!(x.grad(), 12.0, 1e-6);
        assert_approx_eq!(exponent.grad(), 2.0_f64.ln() * 8.0, 1e-6); // dy/dn = ln(x) * x^n
    }

    #[test]
    fn test_pow_backward_variable_exponent() {
        let base = Node::new(3.0);
        let exp = Node::new(2.0);
        let out = base.pow(exp.clone());
        out.backward();

        let expected_base_grad = 2.0 * 3.0_f64.powf(1.0); // 2 * 3^1 = 6
        let expected_exp_grad = 3.0_f64.ln() * 9.0; // ln(3) * 9

        assert_approx_eq!(base.grad(), expected_base_grad, 1e-6);
        assert_approx_eq!(exp.grad(), expected_exp_grad, 1e-6);
    }

    #[test]
    fn test_pow_with_negative_exponent() {
        let base = Node::new(2.0);
        let exponent = Node::new(-1.0);
        let result = base.pow(exponent);
        result.backward();

        // f(x) = x^-1 => f'(x) = -1 * x^-2 = -0.25
        assert_approx_eq!(base.grad(), -0.25, 1e-6);
    }

    #[test]
    fn test_pow_with_zero_exponent() {
        let base = Node::new(5.0);
        let exponent = Node::new(0.0);
        let result = base.pow(exponent.clone());
        result.backward();

        // x^0 = 1, derivative w.r.t base is 0
        assert_approx_eq!(result.data(), 1.0, 1e-6);
        assert_approx_eq!(base.grad(), 0.0, 1e-6);
        // d/dn(x^n) = ln(x) * x^n
        assert_approx_eq!(exponent.grad(), 5.0_f64.ln(), 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let a = Node::new(2.0);
        let b = Node::new(3.0);
        let c = a.clone() * b.clone();
        c.backward();

        assert_eq!(a.grad(), 3.0);
        assert_eq!(b.grad(), 2.0);
        assert_eq!(c.grad(), 1.0);

        c.zero_grad(); //clears all values

        assert_eq!(a.grad(), 0.0);
        assert_eq!(b.grad(), 0.0);
        assert_eq!(c.grad(), 0.0);
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = Node::new(0.0);
        let y = x.sigmoid();
        assert_approx_eq!(y.data(), 0.5, 1e-6);

        let x = Node::new(2.0);
        let y = x.sigmoid();
        let expected = 1.0 / (1.0 + (-2.0f64).exp());
        assert_approx_eq!(y.data(), expected, 1e-6);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = Node::new(0.0);
        let y = x.sigmoid();
        y.backward();

        // Sigmoid at 0 is 0.5, derivative is s * (1 - s) = 0.25
        assert_approx_eq!(x.grad(), 0.25, 1e-6);
    }

    #[test]
    fn test_sigmoid_backward_nonzero_input() {
        let x = Node::new(2.0);
        let y = x.sigmoid();
        y.backward();

        let s = 1.0 / (1.0 + (-2.0f64).exp());
        let expected_grad = s * (1.0 - s);
        assert_approx_eq!(x.grad(), expected_grad, 1e-6);
    }

    #[test]
    fn test_sigmoid_chain_rule() {
        let a = Node::new(1.0);
        let b = a.sigmoid();
        let c = b.clone() * b;
        c.backward();

        // dy/dx = 2s * ds/dx = 2s * s(1 - s) = 2s²(1 - s)
        let s = 1.0 / (1.0 + (-1.0f64).exp());
        let expected_grad = 2.0 * s * s * (1.0 - s);
        assert_approx_eq!(a.grad(), expected_grad, 1e-6);
    }
    #[test]
    fn test_sigmoid_large_negative_input() {
        let x = Node::new(-100.0);
        let y = x.sigmoid();
        y.backward();

        assert_approx_eq!(y.data(), 0.0, 1e-6);
        assert_approx_eq!(x.grad(), 0.0, 1e-6); // gradient should be ~0
    }
}
