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
                Op::Add => {
                    let right = self.pop();
                    let left = self.pop();

                    match (right.borrow(), left.borrow()) {
                        (Object::Int(right), Object::Int(left)) => {
                            let result = right + left;
                            self.push(Rc::new(Object::Int(result)));
                        },
                        _ => panic!("unable to add {:?} and {:?}", right, left),
                    }
                },
                Op::Pop => {
                    self.pop();
                    ()
                },
                _ => panic!("unsupported op {:?}", op),
            }

            ip += 1;
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
        ];

        run_vm_tests(tests);
    }

    fn run_vm_tests(tests: Vec<VMTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let bytecode = compile(program).unwrap();

            use crate::code::InstructionsFns;
            println!("{}", bytecode.instructions.string());

            let mut vm = VM::new(bytecode.constants, bytecode.instructions);

            vm.run();

            let got = vm.last_popped_stack_elem();
            match &got {
                Some(o) => println!("we're good {:?}", o),
                None => println!("not here for {:?}", t.expected),
            }
            test_object(&t.expected, got.unwrap().borrow());
        }
    }

    fn test_object(exp: &Object, got: &Object) {
        match (&exp, &got) {
            (Object::Int(exp), Object::Int(got)) => if exp != got {
                panic!("ints not equal: exp: {}, got: {}", exp, got)
            },
            _ => panic!("can't compare objects: exp: {:?}, got: {:?}", exp, got)
        }
    }
}