use std::cell::{Ref, RefCell};
use std::ops::{Add, Mul, Sub};
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
        self._backward(1.0);
    }

    fn _backward(&self, grad: f64) {
        let mut v = self.0.borrow_mut();
        v.grad += grad;

        match v.op {
            Operator::Add => {
                for child in &v.prev {
                    child._backward(v.grad);
                }
            }
            Operator::Mul => {
                let left = v.prev.get(0).unwrap();
                let right = v.prev.get(1).unwrap();

                let left_data = left.0.borrow().data;
                let right_data = right.0.borrow().data;

                left._backward(right_data * v.grad);
                right._backward(left_data * v.grad);
            }
            Operator::None => {
                // Leaf node, do nothing more
            }
        }
    }
    pub fn pow(self, other: ValueRef) -> ValueRef {
        let data = self.0.borrow().data.powf(other.0.borrow().data);
        let new_value = Value {
            data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Operator::Add,
            //TODO:  this needs to be changed to Operator::Pow and needs to have backwards implemented
        };
        ValueRef(Rc::new(RefCell::new(new_value)))
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
