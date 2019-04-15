use crate::compiler::Compiler;
use crate::object::Object;
use crate::parser::parse;
use crate::code::{Instructions, Op};
use byteorder;
use self::byteorder::{ByteOrder, BigEndian, ReadBytesExt};
use std::rc::Rc;
use std::borrow::Borrow;

const STACK_SIZE: usize = 2048;
const GLOBAL_SIZE: usize = 65536;

pub struct VM<'a> {
    constants: &'a Vec<Rc<Object>>,
    instructions: &'a Instructions,
    stack: Vec<Rc<Object>>,
    sp: usize,
    pub globals: Vec<Rc<Object>>,
}

impl<'a> VM<'a> {
    pub fn new(constants: &'a Vec<Rc<Object>>, instructions: &'a Instructions) -> VM<'a> {
        let mut stack = Vec::with_capacity(STACK_SIZE);
        stack.resize(STACK_SIZE, Rc::new(Object::Null));

        VM{
            constants,
            instructions,
            stack,
            sp: 0,
            globals: VM::new_globals(),
        }
    }

    pub fn new_globals() -> Vec<Rc<Object>> {
        let mut globals = Vec::with_capacity(GLOBAL_SIZE);
        globals.resize(GLOBAL_SIZE, Rc::new(Object::Null));
        globals
    }

    pub fn new_with_global_store(constants: &'a Vec<Rc<Object>>, instructions: &'a Instructions, globals: Vec<Rc<Object>>) -> VM<'a> {
        let mut stack = Vec::with_capacity(STACK_SIZE);
        stack.resize(STACK_SIZE, Rc::new(Object::Null));

        VM{
            constants,
            instructions,
            stack,
            sp: 0,
            globals,
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
                Op::Jump => {
                    let pos = BigEndian::read_u16(&self.instructions[ip+1..ip+3]) as usize;
                    ip = pos - 1;
                },
                Op::JumpNotTruthy => {
                    let pos = BigEndian::read_u16(&self.instructions[ip+1..ip+3]) as usize;
                    ip += 2;

                    let condition = self.pop();
                    if !is_truthy(&condition) {
                        ip = pos - 1;
                    }
                },
                Op::Null => self.push(Rc::new(Object::Null)),
                Op::SetGobal => {
                    let global_index = BigEndian::read_u16(&self.instructions[ip+1..ip+3]) as usize;
                    ip += 2;

                    self.globals[global_index] = self.pop();
                },
                Op::GetGlobal => {
                    let global_index = BigEndian::read_u16(&self.instructions[ip+1..ip+3]) as usize;
                    ip += 2;

                    self.push(Rc::clone(&self.globals[global_index]));
                }
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
                    _ => panic!("unsupported operator in integer binary operation {:?}", op)
                };

                self.push(Rc::new(Object::Int(result)));
            },
            (Object::String(left), Object::String(right)) => {
                let mut result = left.clone();
                match op {
                    Op::Add => result.push_str(&right),
                    _ => panic!("unsupported operator in string binary operation {:?}", op)
                };

                self.push(Rc::new(Object::String(result)));
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
            Object::Null => self.push(Rc::new(Object::Bool(true))),
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

fn is_truthy(obj: &Rc<Object>) -> bool {
    match obj.borrow() {
        Object::Bool(v) => *v,
        Object::Null => false,
        _ => true,
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
            VMTestCase{input: "!(if (false) { 5; })", expected: Object::Bool(true)},
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn conditionals() {
        let tests = vec![
            VMTestCase{input: "if (true) { 10 }", expected: Object::Int(10)},
            VMTestCase{input: "if (true) { 10 } else { 20 }", expected: Object::Int(10)},
            VMTestCase{input: "if (false) { 10 } else { 20 }", expected: Object::Int(20)},
            VMTestCase{input: "if (1) { 10 }", expected: Object::Int(10)},
            VMTestCase{input: "if (1 < 2) { 10 }", expected: Object::Int(10)},
            VMTestCase{input: "if (1 < 2) { 10 } else { 20 }", expected: Object::Int(10)},
            VMTestCase{input: "if (1 > 2) { 10 } else { 20 }", expected: Object::Int(20)},
            VMTestCase{input: "if (1 > 2) { 10 }", expected: Object::Null},
            VMTestCase{input: "if (false) { 10 }", expected: Object::Null},
            VMTestCase{input: "if ((if (false) { 10 })) { 10 } else { 20 }", expected: Object::Int(20)},
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn global_let_statements() {
        let tests = vec![
            VMTestCase{input: "let one = 1; one", expected: Object::Int(1)},
            VMTestCase{input: "let one = 1; let two = 2; one + two", expected: Object::Int(3)},
            VMTestCase{input: "let one = 1; let two = one + one; one + two", expected: Object::Int(3)},
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn string_expressions() {
        let tests = vec![
            VMTestCase{input: "\"monkey\"", expected: Object::String("monkey".to_string())},
            VMTestCase{input: "\"mon\" + \"key\"", expected: Object::String("monkey".to_string())},
            VMTestCase{input: "\"mon\" + \"key\" + \"banana\"", expected: Object::String("monkeybanana".to_string())},
        ];

        run_vm_tests(tests);
    }

    fn run_vm_tests(tests: Vec<VMTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let mut compiler = Compiler::new();
            let bytecode = compiler.compile(program).unwrap();

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
            (Object::String(exp), Object::String(got)) => if exp != got {
                panic!("strings not equal: exp: {}, got: {}", exp, got)
            },
            (Object::Null, Object::Null) => (),
            _ => panic!("can't compare objects: exp: {:?}, got: {:?}", exp, got)
        }
    }
}