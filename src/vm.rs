use crate::compiler::compile;
use crate::object::Object;
use crate::parser::parse;
use crate::code::{Instructions, Op};
use byteorder;
use self::byteorder::{ByteOrder, BigEndian, ReadBytesExt};
use std::rc::Rc;
use std::borrow::Borrow;

const STACK_SIZE: usize = 2048;

struct VM {
    constants: Vec<Rc<Object>>,
    instructions: Instructions,

    stack: Vec<Rc<Object>>,
    sp: usize,
}

impl VM {
    fn stack_top(&self, ) -> Option<Rc<Object>> {
        match self.stack.last() {
            Some(o) => Some(Rc::clone(o)),
            None => None,
        }
    }

    fn run(&mut self) {
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
                _ => panic!("unsupported op {:?}", op),
            }

            ip += 1;
        }
    }

    fn push(&mut self, o: Rc<Object>) {
        if self.sp >= STACK_SIZE {
            panic!("stack overflow")
        }

        self.stack.push(o);
        self.sp += 1;
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
            VMTestCase{input: "1 + 2", expected: Object::Int(2)}, // FIXME
        ];

        run_vm_tests(tests);
    }

    fn run_vm_tests(tests: Vec<VMTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let bytecode = compile(program).unwrap();

            let mut vm = VM{
                constants:bytecode.constants,
                instructions: bytecode.instructions,
                stack: vec![],
                sp: 0,
            };

            vm.run();


            test_object(&t.expected, vm.stack_top().unwrap().borrow());
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