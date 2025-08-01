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
}
