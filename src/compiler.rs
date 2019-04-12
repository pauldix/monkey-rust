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

impl Bytecode {
    fn emit(&mut self, op: Op, operands: &Vec<usize>) -> usize {
        let mut ins = make_instruction(op, operands);
        return self.add_instruction(&mut ins);
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

pub fn compile(node: ast::Node) -> Result {
    let mut bytecode = Bytecode{instructions: vec![], constants: vec![]};

    match node {
        ast::Node::Program(prog) => eval_program(&prog, &mut bytecode)?,
        ast::Node::Statement(stmt) => eval_statement(&stmt, &mut bytecode)?,
        ast::Node::Expression(exp) => eval_expression(&exp, &mut bytecode)?,
    }

    Ok(bytecode)
}

fn eval_program(prog: &ast::Program, bytecode: &mut Bytecode) -> ::std::result::Result<(), CompileError> {
    for stmt in &prog.statements {
        eval_statement(stmt, bytecode)?
    }

    Ok(())
}

fn eval_statement(stmt: &ast::Statement, bytecode: &mut Bytecode) -> ::std::result::Result<(), CompileError> {
    match stmt {
        ast::Statement::Expression(exp) => {
            eval_expression(&exp.expression, bytecode)?;
            // expressions put their value on the stack so this should be popped off since it doesn't get reused
            bytecode.emit(Op::Pop, &vec![]);
        },
        ast::Statement::Return(ret) => panic!("not implemented"),
        ast::Statement::Let(stmt) => panic!("not implemented"),
    }
    Ok(())
}

fn eval_expression(exp: &ast::Expression, bytecode: &mut Bytecode) -> ::std::result::Result<(), CompileError> {
    match exp {
        ast::Expression::Integer(int) => {
            let int = Object::Int(*int);
            let operands = vec![bytecode.add_constant(int)];
            bytecode.emit(Op::Constant, &operands);
        },
        ast::Expression::Boolean(b) => {
            if *b {
                bytecode.emit(Op::True, &vec![]);
            } else {
                bytecode.emit(Op::False, &vec![]);
            }
        },
        ast::Expression::Infix(exp) => {
            if exp.operator == Token::Lt {
                eval_expression(&exp.right, bytecode);
                eval_expression(&exp.left, bytecode);
                bytecode.emit(Op::GreaterThan, &vec![]);
                return Ok(());
            }

            eval_expression(&exp.left, bytecode);
            eval_expression(&exp.right, bytecode);

            match exp.operator {
                Token::Plus => bytecode.emit(Op::Add, &vec![]),
                Token::Minus => bytecode.emit(Op::Sub, &vec![]),
                Token::Asterisk => bytecode.emit(Op::Mul, &vec![]),
                Token::Slash => bytecode.emit(Op::Div, &vec![]),
                Token::Gt => bytecode.emit(Op::GreaterThan, &vec![]),
                Token::Eq => bytecode.emit(Op::Equal, &vec![]),
                Token::Neq => bytecode.emit(Op::NotEqual, &vec![]),
                _ => return Err(CompileError{message: format!("unknown operator {:?}", exp.operator)}),
            };
        },
        ast::Expression::Prefix(exp) => {
            eval_expression(&exp.right, bytecode);

            match exp.operator {
                Token::Minus => bytecode.emit(Op::Minus, &vec![]),
                Token::Bang => bytecode.emit(Op::Bang, &vec![]),
                _ => return Err(CompileError{message: format!("unknown operator {:?}", exp.operator)}),
            };
        },
        _ => panic!("not implemented")
    }

    Ok(())
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

    fn run_compiler_tests(tests: Vec<CompilerTestCase>) {
        for t in tests {
            let program = parse(t.input).unwrap();
            let bytecode = compile(program).unwrap_or_else(
                |err| panic!("{} error compiling on input: {}. want: {:?}", err.message, t.input, t.expected_instructions));

            test_instructions(&t.expected_instructions, bytecode.instructions).unwrap_or_else(
                |err| panic!("{} error on instructions for: {}", &err.message, t.input)
            );

            test_constants(&t.expected_constants, bytecode.constants).unwrap_or_else(
                |err| panic!("{} error on constants for : {}", &err.message, t.input)
            );
        }
    }

    fn test_instructions(expected: &Vec<Instructions>, actual: Instructions) -> ::std::result::Result<(), CompileError> {
        let concatted = concat_instructions(expected);

        if concatted.len() != actual.len() {
            return Err(CompileError{message: format!("instruction lengths not equal\n\texp:\n{:?}\n\tgot:\n{:?}", concatted.string(), actual.string())})
        }

        let mut pos = 0;

        for (exp, got) in concatted.into_iter().zip(actual) {
            if exp != got {
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