#[cfg(test)]
mod tests {
    use rustgrad::Node;

    #[test]
    fn gradients_work_with_adding() {
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
}
