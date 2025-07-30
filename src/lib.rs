use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

#[derive(Clone)]
pub struct Node {
    inner: ValueRef,
}
impl Add for Node {
    type Output = Node;

    fn add(self, other: Node) -> Node {
        let inner = self.inner.clone() + other.inner.clone();
        Node { inner }
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
        self.0.borrow_mut().grad = 1.0;

        let mut stack = vec![self.clone()];

        while let Some(vr) = stack.pop() {
            let v = vr.0.borrow();
            match v.op {
                Operator::Add => {
                    for child in &v.prev {
                        let mut child_mut = child.0.borrow_mut();
                        child_mut.grad += v.grad;
                        stack.push(child.clone());
                    }
                }
                Operator::None => {}
            }
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
