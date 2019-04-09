use crate::object::Object;
use crate::code::{Instructions, InstructionsFns, Op, make_instruction};
use crate::parser::parse;
use crate::ast;
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
  message: String,
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
        ast::Statement::Expression(exp) => eval_expression(&exp.expression, bytecode),
        ast::Statement::Return(ret) => panic!("not implemented"),
        ast::Statement::Let(stmt) => panic!("not implemented"),
    }
}

fn eval_expression(exp: &ast::Expression, bytecode: &mut Bytecode) -> ::std::result::Result<(), CompileError> {
    match exp {
        ast::Expression::Integer(int) => {
            let int = Object::Int(*int);
            let operands = vec![bytecode.add_constant(int)];
            bytecode.emit(Op::Constant, &operands);
        },
        ast::Expression::Infix(exp) => {
            eval_expression(&exp.left, bytecode);
            eval_expression(&exp.right, bytecode);
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
                    make_instruction(Op::Constant, &vec![1])
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