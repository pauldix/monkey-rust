use crate::object::Object;
use crate::code::{Instructions, InstructionsFns, Op, make_instruction};
use crate::parser::parse;
use crate::ast;
use crate::token::Token;
use std::{error, fmt};
use std::fmt::Display;
use std::rc::Rc;
use std::borrow::Borrow;

pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Vec<Rc<Object>>,
}

#[derive(Clone)]
struct EmittedInstruction {
    op_code: Op,
    position: usize,
}

impl EmittedInstruction {
    fn is_pop(&self) -> bool {
        match self.op_code {
            Op::Pop => true,
            _ => false,
        }
    }
}

pub struct Compiler {
    pub instructions: Instructions,
    pub constants: Vec<Rc<Object>>,

    last_instruction: Option<EmittedInstruction>,
    previous_instruction: Option<EmittedInstruction>,
}

impl Compiler {
    pub fn new() -> Compiler {
        Compiler{
            instructions: vec![],
            constants: vec![],
            last_instruction: None,
            previous_instruction: None,
        }
    }

    pub fn compile(&mut self, node: ast::Node) -> Result {
        match node {
            ast::Node::Program(prog) => self.eval_program(&prog)?,
            ast::Node::Statement(stmt) => self.eval_statement(&stmt)?,
            ast::Node::Expression(exp) => self.eval_expression(&exp)?,
        }

        Ok(self.bytecode())
    }

    pub fn bytecode(&mut self) -> Bytecode {
        // TODO: figure out how to do something like this without cloning
        Bytecode{
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
        }
    }

    fn emit(&mut self, op: Op, operands: &Vec<usize>) -> usize {
        let mut ins = make_instruction(op.clone(), &operands);
        let pos= self.add_instruction(&mut ins);
        self.set_last_instruction(op, pos);

        return pos;
    }

    fn set_last_instruction(&mut self, op_code: Op, position: usize) {
        match &self.last_instruction {
            Some(ins) => self.previous_instruction = Some(ins.clone()),
            _ => (),
        }
        self.last_instruction = Some(EmittedInstruction{op_code, position});
    }

    fn add_instruction(&mut self, ins: &Vec<u8>) -> usize {
        let pos = self.instructions.len();
        self.instructions.extend_from_slice(ins);
        return pos;
    }

    fn add_constant(&mut self, obj: Object) -> usize {
        self.constants.push(Rc::new(obj));
        return self.constants.len() - 1;
    }

    fn eval_program(&mut self, prog: &ast::Program) -> ::std::result::Result<(), CompileError> {
        for stmt in &prog.statements {
            self.eval_statement(stmt)?
        }

        Ok(())
    }

    fn eval_statement(&mut self, stmt: &ast::Statement) -> ::std::result::Result<(), CompileError> {
        match stmt {
            ast::Statement::Expression(exp) => {
                self.eval_expression(&exp.expression)?;
                // expressions put their value on the stack so this should be popped off since it doesn't get reused
                self.emit(Op::Pop, &vec![]);
            },
            ast::Statement::Return(ret) => panic!("not implemented"),
            ast::Statement::Let(stmt) => panic!("not implemented"),
        }
        Ok(())
    }

    fn eval_block_statement(&mut self, stmt: &ast::BlockStatement) -> ::std::result::Result<(), CompileError> {
        for stmt in &stmt.statements {
            self.eval_statement(stmt)?;
        }

        Ok(())
    }

    fn eval_expression(&mut self, exp: &ast::Expression) -> ::std::result::Result<(), CompileError> {
        match exp {
            ast::Expression::Integer(int) => {
                let int = Object::Int(*int);
                let operands = vec![self.add_constant(int)];
                self.emit(Op::Constant, &operands);
            },
            ast::Expression::Boolean(b) => {
                if *b {
                    self.emit(Op::True, &vec![]);
                } else {
                    self.emit(Op::False, &vec![]);
                }
            },
            ast::Expression::Infix(exp) => {
                if exp.operator == Token::Lt {
                    self.eval_expression(&exp.right);
                    self.eval_expression(&exp.left);
                    self.emit(Op::GreaterThan, &vec![]);
                    return Ok(());
                }

                self.eval_expression(&exp.left);
                self.eval_expression(&exp.right);

                match exp.operator {
                    Token::Plus => self.emit(Op::Add, &vec![]),
                    Token::Minus => self.emit(Op::Sub, &vec![]),
                    Token::Asterisk => self.emit(Op::Mul, &vec![]),
                    Token::Slash => self.emit(Op::Div, &vec![]),
                    Token::Gt => self.emit(Op::GreaterThan, &vec![]),
                    Token::Eq => self.emit(Op::Equal, &vec![]),
                    Token::Neq => self.emit(Op::NotEqual, &vec![]),
                    _ => return Err(CompileError{message: format!("unknown operator {:?}", exp.operator)}),
                };
            },
            ast::Expression::Prefix(exp) => {
                self.eval_expression(&exp.right);

                match exp.operator {
                    Token::Minus => self.emit(Op::Minus, &vec![]),
                    Token::Bang => self.emit(Op::Bang, &vec![]),
                    _ => return Err(CompileError{message: format!("unknown operator {:?}", exp.operator)}),
                };
            },
            ast::Expression::If(ifexp) => {
                self.eval_expression(&ifexp.condition);

                let jump_not_truthy_pos = self.emit(Op::JumpNotTruthy, &vec![9999]);

                self.eval_block_statement(&ifexp.consequence);

                if let Some(ins) = &self.last_instruction {
                    if ins.is_pop() {
                        self.remove_last_instruction();
                    }
                }

                // set the jump position
                let mut after_consequence;

                if let Some(alternative) = &ifexp.alternative {
                    let jump_pos = self.emit(Op::Jump, &vec![9999]);

                    after_consequence = self.instructions.len();
                    self.change_operand(jump_not_truthy_pos, after_consequence);

                    self.eval_block_statement(alternative)?;

                    if let Some(ins) = &self.last_instruction {
                        if ins.is_pop() {
                            self.remove_last_instruction();
                        }
                    }

                    let after_alternative_pos = self.instructions.len();
                    self.change_operand(jump_pos, after_alternative_pos);
                } else {
                    after_consequence = self.instructions.len();
                    self.change_operand(jump_not_truthy_pos, after_consequence);
                }
            },
            _ => panic!("not implemented")
        }

        Ok(())
    }

    fn remove_last_instruction(&mut self) {
        let pos = match &self.last_instruction {
            Some(ins) => ins.position,
            _ => 0,
        };

        self.instructions.truncate(pos);
        self.last_instruction = self.previous_instruction.clone();
    }

    fn replace_instruction(&mut self, pos: usize, ins: &[u8]) {
        let mut i = 0;
        while i < ins.len() {
            self.instructions[pos + i] = ins[i];
            i += 1;
        }
    }

    fn change_operand(&mut self, pos: usize, operand: usize) {
        let op = unsafe { ::std::mem::transmute(self.instructions[pos]) };
        let ins = make_instruction(op, &vec![operand]);
        self.replace_instruction(pos, &ins);
    }
}

type Result = ::std::result::Result<Bytecode, CompileError>;

#[derive(Debug)]
pub struct CompileError {
  pub message: String,
}

impl error::Error for CompileError {
    fn description(&self) -> &str { &self.message }
}

impl Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompileError: {}", &self.message)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct CompilerTestCase<'a> {
        input: &'a str,
        expected_constants: Vec<Object>,
        expected_instructions: Vec<Instructions>,
    }

    #[test]
    fn integer_arithmetic() {
        let tests = vec![
            CompilerTestCase{
                input:"1 + 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Add, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input:"1; 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Pop, &vec![]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 - 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Sub, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 * 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Mul, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "2 / 1",
                expected_constants: vec![Object::Int(2), Object::Int(1)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Div, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "-1",
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Minus, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
        ];

        run_compiler_tests(tests)
    }

    #[test]
    fn boolean_expressions() {
        let tests = vec![
            CompilerTestCase{
                input: "true",
                expected_constants: vec![],
                expected_instructions: vec![
                    make_instruction(Op::True, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "false",
                expected_constants: vec![],
                expected_instructions: vec![
                    make_instruction(Op::False, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 > 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::GreaterThan, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 < 2",
                expected_constants: vec![Object::Int(2), Object::Int(1)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::GreaterThan, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 == 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::Equal, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "1 != 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make_instruction(Op::Constant, &vec![0]),
                    make_instruction(Op::Constant, &vec![1]),
                    make_instruction(Op::NotEqual, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "true == false",
                expected_constants: vec![],
                expected_instructions: vec![
                    make_instruction(Op::True, &vec![]),
                    make_instruction(Op::False, &vec![]),
                    make_instruction(Op::Equal, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "true != false",
                expected_constants: vec![],
                expected_instructions: vec![
                    make_instruction(Op::True, &vec![]),
                    make_instruction(Op::False, &vec![]),
                    make_instruction(Op::NotEqual, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "!true",
                expected_constants: vec![],
                expected_instructions: vec![
                    make_instruction(Op::True, &vec![]),
                    make_instruction(Op::Bang, &vec![]),
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
        ];

        run_compiler_tests(tests)
    }

    #[test]
    fn conditionals() {
        let tests = vec![
            CompilerTestCase{
                input: "if (true) { 10 }; 3333;",
                expected_constants: vec![Object::Int(10), Object::Int(3333)],
                expected_instructions: vec![
                    // 0000
                    make_instruction(Op::True, &vec![]),
                    // 0001
                    make_instruction(Op::JumpNotTruthy, &vec![7]),
                    // 0004
                    make_instruction(Op::Constant, &vec![0]),
                    // 0007
                    make_instruction(Op::Pop, &vec![]),
                    // 0008
                    make_instruction(Op::Constant, &vec![1]),
                    // 0011
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
            CompilerTestCase{
                input: "if (true) { 10 } else { 20 }; 3333;",
                expected_constants: vec![Object::Int(10), Object::Int(20), Object::Int(3333)],
                expected_instructions: vec![
                    // 0000
                    make_instruction(Op::True, &vec![]),
                    // 0001
                    make_instruction(Op::JumpNotTruthy, &vec![10]),
                    // 0004
                    make_instruction(Op::Constant, &vec![0]),
                    // 0007
                    make_instruction(Op::Jump, &vec![13]),
                    // 0010
                    make_instruction(Op::Constant, &vec![1]),
                    // 0013
                    make_instruction(Op::Pop, &vec![]),
                    // 0014
                    make_instruction(Op::Constant, &vec![2]),
                    // 0017
                    make_instruction(Op::Pop, &vec![]),
                ],
            },
        ];

        run_compiler_tests(tests)
    }

    fn run_compiler_tests(tests: Vec<CompilerTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let mut compiler = Compiler::new();
            let bytecode = compiler.compile(program).unwrap_or_else(
                |err| panic!("{} error compiling on input: {}. want: {:?}", err.message, t.input, t.expected_instructions));

            test_instructions(&t.expected_instructions, &bytecode.instructions).unwrap_or_else(
                |err| panic!("{} error on instructions for: {}\nexp: {}\ngot: {}", &err.message, t.input, concat_instructions(&t.expected_instructions).string(), bytecode.instructions.string())
            );

            test_constants(&t.expected_constants, bytecode.constants).unwrap_or_else(
                |err| panic!("{} error on constants for : {}", &err.message, t.input)
            );
        }
    }


    fn test_instructions(expected: &Vec<Instructions>, actual: &Instructions) -> ::std::result::Result<(), CompileError> {
        let concatted = concat_instructions(expected);

        if concatted.len() != actual.len() {
            return Err(CompileError{message: format!("instruction lengths not equal\n\texp:\n{:?}\n\tgot:\n{:?}", concatted.string(), actual.string())})
        }

        let mut pos = 0;

        for (exp, got) in concatted.into_iter().zip(actual) {
            if exp != *got {
                return Err(CompileError { message: format!("exp\n{:?} but got\n{} at position {:?}", exp, got, pos) });
            }
            pos = pos + 1;
        }
        Ok(())
    }

    fn test_constants(expected: &Vec<Object>, actual: Vec<Rc<Object>>) -> ::std::result::Result<(), CompileError> {
        let mut pos = 0;

        for (exp, got) in expected.into_iter().zip(actual) {
            let got = got.borrow();
            match (exp, got) {
                (Object::Int(exp), Object::Int(got)) => if *exp != *got {
                    return Err(CompileError{message: format!("constant {}, exp: {} got: {}", pos, exp, got)})
                },
                _ => panic!("can't compare objects: exp: {:?} got: {:?}", exp, got)
            }
            pos = pos + 1;
        }
        Ok(())
    }

    fn concat_instructions(instructions: &Vec<Instructions>) -> Instructions {
        let mut concatted = Instructions::new();

        for i in instructions {
            for u in i {
                concatted.push(*u);
            }
        }

        concatted
    }
}