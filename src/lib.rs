use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;
#[derive(Clone)]
pub struct Node {
    inner: ValueRef,
}
impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        Node {
            inner: self.inner.clone() + other.inner.clone(),
        }
    }
}

impl Sub for Node {
    type Output = Node;
    fn sub(self, other: Self) -> Self::Output {
        Node {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }
}
impl Mul for Node {
    type Output = Node;
    fn mul(self, other: Node) -> Node {
        Node {
            inner: self.inner.clone() * other.inner.clone(),
        }
    }
}
impl Div for Node {
    type Output = Node;
    fn div(self, other: Node) -> Node {
        Node {
            inner: self.inner.clone() / other.inner.clone(),
        }
    }
}

impl Neg for Node {
    type Output = Node;
    fn neg(self) -> Node {
        Node {
            inner: -self.inner.clone(),
        }
    }
}

impl Node {
    pub fn new(data: f64) -> Self {
        Node {
            inner: ValueRef::new(data),
        }
    }

    pub fn backward(&self) {
        self.inner.clone().backward();
    }

    pub fn data(&self) -> f64 {
        self.inner.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.0.borrow().grad
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Operator {
    None,
    Add,
    Mul,
    Pow,
    Neg,
    Div,
}

#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<ValueRef>,
    pub op: Operator,
}

#[derive(Clone, Debug)]
pub struct ValueRef(pub Rc<RefCell<Value>>);

impl ValueRef {
    pub fn new(data: f64) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            prev: vec![],
            op: Operator::None,
        })))
    }

    pub fn backward(self) {
        let mut visited = std::collections::HashSet::new();
        let mut order = vec![];

        fn topo_sort(
            v: &ValueRef,
            visited: &mut std::collections::HashSet<*const RefCell<Value>>,
            order: &mut Vec<ValueRef>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for child in &v.0.borrow().prev {
                topo_sort(child, visited, order);
            }
            order.push(v.clone());
        }

        topo_sort(&self, &mut visited, &mut order);

        self.0.borrow_mut().grad = 1.0;

        for node in order.into_iter().rev() {
            node._backward();
        }
    }

    fn _backward(&self) {
        let v = self.0.borrow();

        match v.op {
            Operator::Add => {
                for child in &v.prev {
                    child.0.borrow_mut().grad += v.grad;
                }
            }
            Operator::Mul => {
                let left = &v.prev[0];
                let right = &v.prev[1];

                let left_val = left.0.borrow().data;
                let right_val = right.0.borrow().data;

                left.0.borrow_mut().grad += right_val * v.grad;
                right.0.borrow_mut().grad += left_val * v.grad;
            }
            Operator::Pow => {
                let base = &v.prev[0];
                let exponent = &v.prev[1];

                let base_val = base.0.borrow().data;
                let exponent_val = exponent.0.borrow().data;

                let base_grad = exponent_val * base_val.powf(exponent_val - 1.0) * v.grad;
                let exponent_grad = base_val.ln() * base_val.powf(exponent_val) * v.grad;

                base.0.borrow_mut().grad += base_grad;
                exponent.0.borrow_mut().grad += exponent_grad;
            }
            Operator::Div => {
                let left = &v.prev[0];
                let right = &v.prev[1];

                let left_val = left.0.borrow().data;
                let right_val = right.0.borrow().data;

                left.0.borrow_mut().grad += (1.0 / right_val) * v.grad;
                right.0.borrow_mut().grad += (-left_val / (right_val * right_val)) * v.grad;
            }
            Operator::Neg => {
                let x = &v.prev[0];
                x.0.borrow_mut().grad += -v.grad;
            }
            Operator::None => {}
        }
    }
}

impl Add for ValueRef {
    type Output = ValueRef;

    fn add(self, other: ValueRef) -> ValueRef {
        let data = self.0.borrow().data + other.0.borrow().data;

        let new_value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Operator::Add,
        };

        ValueRef(Rc::new(RefCell::new(new_value)))
    }
}

impl Sub for ValueRef {
    type Output = ValueRef;
    fn sub(self, rhs: Self) -> ValueRef {
        let data = self.0.borrow().data - rhs.0.borrow().data;
        let new_value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone(), rhs.clone()],
            op: Operator::Add,
        };
        ValueRef(Rc::new(RefCell::new(new_value)))
    }
}

impl Mul for ValueRef {
    type Output = ValueRef;
    fn mul(self, other: Self) -> ValueRef {
        let data = self.0.borrow().data * other.0.borrow().data;
        let new_value: Value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Operator::Mul,
        };
        ValueRef(Rc::new(RefCell::new(new_value)))
    }
}

impl Neg for ValueRef {
    type Output = ValueRef;
    fn neg(self) -> ValueRef {
        let data = -self.0.borrow().data;
        let new_value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone()],
            op: Operator::Neg,
        };
        ValueRef(Rc::new(RefCell::new(new_value)))
    }
}

impl Div for ValueRef {
    type Output = ValueRef;
    fn div(self, other: Self) -> ValueRef {
        let data = self.0.borrow().data / other.0.borrow().data;
        let new_value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Operator::Div,
        };
        ValueRef(Rc::new(RefCell::new(new_value)))
    }
}
