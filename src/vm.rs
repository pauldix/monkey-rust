use crate::compiler::Compiler;
use crate::object::{Object, Array, MonkeyHash, CompiledFunction};
use crate::parser::parse;
use crate::code::{Instructions, Op};
use byteorder;
use self::byteorder::{ByteOrder, BigEndian, ReadBytesExt};
use std::rc::Rc;
use std::borrow::Borrow;
use std::collections::HashMap;

const STACK_SIZE: usize = 2048;
const GLOBAL_SIZE: usize = 65536;
const MAX_FRAMES: usize = 1024;

#[derive(Clone)]
struct Frame {
    func: Rc<CompiledFunction>,
    ip: usize,
}

//impl Frame {
//    fn instructions(&mut self) -> &mut Instructions {
//        &mut self.func.instructions
//    }
//}

pub struct VM<'a> {
    constants: &'a Vec<Rc<Object>>,

    stack: Vec<Rc<Object>>,
    sp: usize,

    pub globals: Vec<Rc<Object>>,

    frames: Vec<Frame>,
    frames_index: usize,
}

impl<'a> VM<'a> {
    pub fn new(constants: &'a Vec<Rc<Object>>, instructions: Instructions) -> VM<'a> {
        let mut stack = Vec::with_capacity(STACK_SIZE);
        stack.resize(STACK_SIZE, Rc::new(Object::Null));

        let mut frames = Vec::with_capacity(MAX_FRAMES);
        frames.resize(MAX_FRAMES, Frame{func: Rc::new(CompiledFunction{instructions: vec![]}), ip: 0});

        let main_func = Rc::new(CompiledFunction{instructions});
        let main_frame = Frame{func: main_func, ip: 0};
        frames.push(main_frame);

        VM{
            constants,
            stack,
            sp: 0,
            globals: VM::new_globals(),
            frames,
            frames_index: 1,
        }
    }

    pub fn new_globals() -> Vec<Rc<Object>> {
        let mut globals = Vec::with_capacity(GLOBAL_SIZE);
        globals.resize(GLOBAL_SIZE, Rc::new(Object::Null));
        return globals;
    }

    pub fn new_with_global_store(constants: &'a Vec<Rc<Object>>, instructions: Instructions, globals: Vec<Rc<Object>>) -> VM<'a> {
        let mut vm = VM::new(constants, instructions);
        vm.globals = globals;
        return vm;
    }

    pub fn last_popped_stack_elem(&self, ) -> Option<Rc<Object>> {
        match self.stack.get(self.sp) {
            Some(o) => Some(Rc::clone(o)),
            None => None,
        }
    }

    fn current_instructions_len(&mut self) -> usize {
        self.frames.last().unwrap().func.instructions.len()
    }

    pub fn run(&mut self) {
        let mut ip = 0;

        while ip < self.current_instructions_len() - 1 {
            let frame = self.frames.last().unwrap();
            ip = frame.ip;
            let ins = &frame.func.instructions;
            let op = unsafe { ::std::mem::transmute(*ins.get_unchecked(ip)) };

            match op {
                Op::Constant => {
                    let const_index = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
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
                    let pos = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip = pos - 1;
                },
                Op::JumpNotTruthy => {
                    let pos = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip += 2;

                    let condition = self.pop();
                    if !is_truthy(&condition) {
                        ip = pos - 1;
                    }
                },
                Op::Null => self.push(Rc::new(Object::Null)),
                Op::SetGobal => {
                    let global_index = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip += 2;

                    self.globals[global_index] = self.pop();
                },
                Op::GetGlobal => {
                    let global_index = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip += 2;

                    self.push(Rc::clone(&self.globals[global_index]));
                },
                Op::Array => {
                    let num_elements = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip += 2;

                    let array = self.build_array(self.sp - num_elements, self.sp);
                    self.sp -= num_elements;

                    self.push(Rc::new(array));
                },
                Op::Hash => {
                    let num_elements = BigEndian::read_u16(&ins[ip+1..ip+3]) as usize;
                    ip += 2;

                    let hash = self.build_hash(self.sp - num_elements, self.sp);
                    self.sp -= num_elements;

                    self.push(Rc::new(hash));
                },
                Op::Index => {
                    let index = self.pop();
                    let left = self.pop();

                    self.execute_index_expression(left, index);
                },
                _ => panic!("unsupported op {:?}", op),
            }

            self.frames.last_mut().unwrap().ip = ip + 1;
        }
    }

    fn execute_index_expression(&mut self, left: Rc<Object>, index: Rc<Object>) {
        match (&*left, &*index) {
            (Object::Array(arr), Object::Int(idx)) => self.execute_array_index(&arr, *idx),
            (Object::Hash(hash), _) => self.execute_hash_index(hash, index),
            _ => panic!("index not supported on: {:?} {:?}", left, index),
        }
    }

    fn execute_array_index(&mut self, arr: &Rc<Array>, idx: i64) {
        match arr.elements.get(idx as usize) {
            Some(el) => self.push(Rc::clone(el)),
            None => self.push(Rc::new(Object::Null)),
        }
    }

    fn execute_hash_index(&mut self, hash: &Rc<MonkeyHash>, index: Rc<Object>) {
        match &*index {
            Object::String(_) | Object::Int(_) | Object::Bool(_) => {
                match hash.pairs.get(&*index) {
                    Some(obj) => self.push(Rc::clone(obj)),
                    None => self.push(Rc::new(Object::Null)),
                }
            },
            _ => panic!("unusable as hash key: {}", index)
        }
    }

    fn build_array(&mut self, start_index: usize, end_index: usize) -> Object {
        let mut elements = Vec::with_capacity(end_index - start_index);
        elements.resize(end_index - start_index, Rc::new(Object::Null));
        let mut i = start_index;

        while i < end_index {
            elements[i-start_index] = self.stack[i].clone();
            i += 1;
        }

        Object::Array(Rc::new(Array{elements}))
    }

    fn build_hash(&mut self, start_index: usize, end_index: usize) -> Object {
        let mut hash = HashMap::new();
        let mut i = start_index;

        while i < end_index {
            let key = self.stack[i].clone();
            let value = self.stack[i + 1].clone();

            hash.insert(key, value);
            i += 2;
        }

        Object::Hash(Rc::new(MonkeyHash{pairs: hash}))
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
    use std::collections::HashMap;

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

    #[test]
    fn array_literals() {
        let tests = vec![
            VMTestCase{
                input: "[]",
                expected: Object::Array(Rc::new(Array{
                    elements: vec![],
                })),
            },
            VMTestCase{
                input: "[1, 2, 3]",
                expected: Object::Array(Rc::new(Array{
                    elements: vec![
                        Rc::new(Object::Int(1)),
                        Rc::new(Object::Int(2)),
                        Rc::new(Object::Int(3)),
                    ],
                })),
            },
            VMTestCase{
                input: "[1 + 2, 3 * 4, 5 + 6]",
                expected: Object::Array(Rc::new(Array{
                    elements: vec![
                        Rc::new(Object::Int(3)),
                        Rc::new(Object::Int(12)),
                        Rc::new(Object::Int(11)),
                    ],
                })),
            },
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn hash_literals() {
        macro_rules! map(
            { $($key:expr => $value:expr),+ } => {
                {
                    let mut m = ::std::collections::HashMap::new();
                    $(
                        m.insert($key, $value);
                    )+
                    m
                }
            };
        );

        let tests = vec![
            VMTestCase{
                input: "{}",
                expected: hash_to_object(HashMap::new()),
            },
            VMTestCase{
                input: "{1: 2, 2: 3}",
                expected: hash_to_object(map!{1 => 2, 2 => 3}),
            },
            VMTestCase{
                input: "{1 + 1: 2 * 2, 3 + 3: 4 * 4}",
                expected: hash_to_object(map!{2 => 4, 6 => 16}),
            },
        ];

        run_vm_tests(tests);
    }

    #[test]
    fn index_expressions() {
        let tests = vec![
            VMTestCase{input: "[1, 2, 3][1]", expected: Object::Int(2)},
            VMTestCase{input: "[1, 2, 3][0 + 2]", expected: Object::Int(3)},
            VMTestCase{input: "[[1, 1, 1]][0][0]", expected: Object::Int(1)},
            VMTestCase{input: "[][0]", expected: Object::Null},
            VMTestCase{input: "[1, 2, 3][99]", expected: Object::Null},
            VMTestCase{input: "[1][-1]", expected: Object::Null},
            VMTestCase{input: "{1: 1, 2: 2}[1]", expected: Object::Int(1)},
            VMTestCase{input: "{1: 1, 2: 2}[2]", expected: Object::Int(2)},
            VMTestCase{input: "{1: 1}[0]", expected: Object::Null},
            VMTestCase{input: "{}[0]", expected: Object::Null},
        ];

        run_vm_tests(tests);
    }

    fn hash_to_object(h: HashMap<i64,i64>) -> Object {
        let hash = HashMap::new();
        let mut mh = MonkeyHash{pairs: hash};

        for (h, k) in h {
            mh.pairs.insert(Rc::new(Object::Int(h)), Rc::new(Object::Int(k)));
        }

        Object::Hash(Rc::new(mh))
    }

    fn run_vm_tests(tests: Vec<VMTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let mut compiler = Compiler::new();
            let bytecode = compiler.compile(program).unwrap();

            let mut vm = VM::new(bytecode.constants, bytecode.instructions.to_vec());

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
            (Object::Array(exp), Object::Array(got)) => if exp != got {
                panic!("arrays not equal: exp: {:?}, got: {:?}", exp, got)
            },
            (Object::Hash(exp), Object::Hash(got)) => if exp != got {
                panic!("hashes not equal: exp: {:?}, got: {:?}", exp, got)
            },
            (Object::Null, Object::Null) => (),
            _ => panic!("can't compare objects: exp: {:?}, got: {:?}", exp, got)
        }
    }
}