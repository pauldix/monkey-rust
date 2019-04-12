use crate::compiler::compile;
use crate::object::Object;
use crate::parser::parse;
use crate::code::{Instructions, Op};
use byteorder;
use self::byteorder::{ByteOrder, BigEndian, ReadBytesExt};
use std::rc::Rc;
use std::borrow::Borrow;

const STACK_SIZE: usize = 2048;

pub struct VM {
    pub constants: Vec<Rc<Object>>,
    pub instructions: Instructions,

    pub stack: Vec<Rc<Object>>,
    pub sp: usize,
}

impl VM {
    pub fn new(constants: Vec<Rc<Object>>, instructions: Instructions) -> VM {
        let mut stack = Vec::with_capacity(STACK_SIZE);
        stack.resize(STACK_SIZE, Rc::new(Object::Null));

        VM{
            constants,
            instructions,
            stack,
            sp: 0,
        }
    }

    pub fn last_popped_stack_elem(&self, ) -> Option<Rc<Object>> {
        match self.stack.get(self.sp) {
            Some(o) => Some(Rc::clone(o)),
            None => None,
        }
    }

    pub fn run(&mut self) {
        let mut ip = 0;

        while ip < self.instructions.len() {
            let op = unsafe { ::std::mem::transmute(**&self.instructions.get_unchecked(ip)) };

            match op {
                Op::Constant => {
                    let const_index = BigEndian::read_u16(&self.instructions[ip+1..ip+3]) as usize;
                    ip += 2;

                    let c = Rc::clone(self.constants.get(const_index).unwrap());
                    self.push(c)
                },
                Op::Add | Op::Sub | Op::Mul | Op::Div => self.execute_binary_operation(op),
                Op::GreaterThan | Op::Equal | Op::NotEqual => self.execute_comparison(op),
                Op::Pop => {
                    self.pop();
                    ()
                },
                Op::True => self.push(Rc::new(Object::Bool(true))),
                Op::False => self.push(Rc::new(Object::Bool(false))),
                Op::Bang => self.execute_bang_operator(),
                Op::Minus => self.execute_minus_operator(),
                _ => panic!("unsupported op {:?}", op),
            }

            ip += 1;
        }
    }

    fn execute_binary_operation(&mut self, op: Op) {
        let right = self.pop();
        let left = self.pop();

        match (left.borrow(), right.borrow()) {
            (Object::Int(left), Object::Int(right)) => {
                let result = match op {
                    Op::Add => left + right,
                    Op::Sub => left - right,
                    Op::Mul => left * right,
                    Op::Div => left / right,
                    _ => panic!("unsupported operator in binary operation {:?}", op)
                };

                self.push(Rc::new(Object::Int(result)));
            },
            _ => panic!("unable to {:?} {:?} and {:?}", op, left, right),
        }
    }

    fn execute_comparison(&mut self, op: Op) {
        let right = self.pop();
        let left = self.pop();

        match (left.borrow(), right.borrow()) {
            (Object::Int(left), Object::Int(right)) => {
                let result = match op {
                    Op::Equal => left == right,
                    Op::NotEqual => left != right,
                    Op::GreaterThan => left > right,
                    _ => panic!("unsupported operator in comparison {:?}", op)
                };

                self.push(Rc::new(Object::Bool(result)));
            },
            (Object::Bool(left), Object::Bool(right)) => {
                let result = match op {
                    Op::Equal => left == right,
                    Op::NotEqual => left != right,
                    _ => panic!("unsupported operator in comparison {:?}", op)
                };

                self.push(Rc::new(Object::Bool(result)));
            },
            _ => panic!("unable to {:?} {:?} and {:?}", op, left, right),
        }
    }

    fn execute_bang_operator(&mut self) {
        let op = self.pop();

        match op.borrow() {
            Object::Bool(true) => self.push(Rc::new(Object::Bool(false))),
            Object::Bool(false) => self.push(Rc::new(Object::Bool(true))),
            _ => self.push(Rc::new(Object::Bool(false))),
        }
    }

    fn execute_minus_operator(&mut self) {
        let op = self.pop();

        match op.borrow() {
            Object::Int(int) => self.push(Rc::new(Object::Int(-*int))),
            _ => panic!("unsupported type for negation {:?}", op)
        }
    }

    fn push(&mut self, o: Rc<Object>) {
        if self.sp >= STACK_SIZE {
            panic!("stack overflow")
        }

        self.stack[self.sp] = o;
        self.sp += 1;
    }

    fn pop(&mut self) -> Rc<Object> {
        self.sp -= 1;
        Rc::clone(&self.stack[self.sp])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::rc::Rc;

    struct VMTestCase<'a> {
        input: &'a str,
        expected: Object,
    }

    #[test]
    fn integer_arithmetic() {
        let tests = vec![
            VMTestCase{input: "1", expected: Object::Int(1)},
            VMTestCase{input: "2", expected: Object::Int(2)},
            VMTestCase{input: "1 + 2", expected: Object::Int(3)},
            VMTestCase{input: "1 - 2", expected: Object::Int(-1)},
            VMTestCase{input: "1 * 2", expected: Object::Int(2)},
            VMTestCase{input: "4 / 2", expected: Object::Int(2)},
            VMTestCase{input: "50 / 2 * 2 + 10 - 5", expected: Object::Int(55)},
            VMTestCase{input: "5 * (2 + 10)", expected: Object::Int(60)},
            VMTestCase{input: "5 + 5 + 5 + 5 - 10", expected: Object::Int(10)},
            VMTestCase{input: "2 * 2 * 2 * 2 * 2", expected: Object::Int(32)},
            VMTestCase{input: "5 * 2 + 10", expected: Object::Int(20)},
            VMTestCase{input: "5 + 2 * 10", expected: Object::Int(25)},
            VMTestCase{input: "5 * (2 + 10)", expected: Object::Int(60)},
            VMTestCase{input: "-5", expected: Object::Int(-5)},
            VMTestCase{input: "-10", expected: Object::Int(-10)},
            VMTestCase{input: "-50 + 100 + -50", expected: Object::Int(0)},
            VMTestCase{input: "(5 + 10 * 2 + 15 / 3) * 2 + -10", expected: Object::Int(50)},
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn boolean_expressions() {
        let tests = vec![
            VMTestCase{input: "true", expected: Object::Bool(true)},
            VMTestCase{input: "false", expected: Object::Bool(false)},
            VMTestCase{input: "1 < 2", expected: Object::Bool(true)},
            VMTestCase{input: "1 > 2", expected: Object::Bool(false)},
            VMTestCase{input: "1 < 1", expected: Object::Bool(false)},
            VMTestCase{input: "1 > 1", expected: Object::Bool(false)},
            VMTestCase{input: "1 == 1", expected: Object::Bool(true)},
            VMTestCase{input: "1 != 1", expected: Object::Bool(false)},
            VMTestCase{input: "1 == 2", expected: Object::Bool(false)},
            VMTestCase{input: "1 != 2", expected: Object::Bool(true)},
            VMTestCase{input: "true == true", expected: Object::Bool(true)},
            VMTestCase{input: "false == false", expected: Object::Bool(true)},
            VMTestCase{input: "true == false", expected: Object::Bool(false)},
            VMTestCase{input: "true != false", expected: Object::Bool(true)},
            VMTestCase{input: "false != true", expected: Object::Bool(true)},
            VMTestCase{input: "(1 < 2) == true", expected: Object::Bool(true)},
            VMTestCase{input: "(1 < 2) == false", expected: Object::Bool(false)},
            VMTestCase{input: "(1 > 2) == true", expected: Object::Bool(false)},
            VMTestCase{input: "(1 > 2) == false", expected: Object::Bool(true)},
            VMTestCase{input: "!true", expected: Object::Bool(false)},
            VMTestCase{input: "!false", expected: Object::Bool(true)},
            VMTestCase{input: "!5", expected: Object::Bool(false)},
            VMTestCase{input: "!!true", expected: Object::Bool(true)},
            VMTestCase{input: "!!false", expected: Object::Bool(false)},
            VMTestCase{input: "!!5", expected: Object::Bool(true)},
        ];

        run_vm_tests(tests);
    }

    fn run_vm_tests(tests: Vec<VMTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let bytecode = compile(program).unwrap();

            let mut vm = VM::new(bytecode.constants, bytecode.instructions);

            vm.run();

            let got = vm.last_popped_stack_elem();
            test_object(&t.expected, got.unwrap().borrow());
        }
    }

    fn test_object(exp: &Object, got: &Object) {
        match (&exp, &got) {
            (Object::Int(exp), Object::Int(got)) => if exp != got {
                panic!("ints not equal: exp: {}, got: {}", exp, got)
            },
            (Object::Bool(exp), Object::Bool(got)) => if exp != got {
                panic!("bools not equal: exp: {}, got: {}", exp, got)
            },
            _ => panic!("can't compare objects: exp: {:?}, got: {:?}", exp, got)
        }
    }
}