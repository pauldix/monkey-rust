use crate::object::Object;
use crate::code::{Instructions, InstructionsFns, Op, make_instruction};
use crate::parser::parse;
use crate::ast;
use std::{error, fmt};
use std::fmt::Display;

struct Bytecode {
    instructions: Instructions,
    constants: Vec<Object>,
}

type Result = ::std::result::Result<Bytecode, CompileError>;

#[derive(Debug)]
struct CompileError {
  message: String,
}

impl error::Error for CompileError {
    fn description(&self) -> &str { &self.message }
}

impl Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CompileError: {}", &self.message)
    }
}

fn compile(node: ast::Node) -> Result {
    match node {
        ast::Node::Program(prog) => eval_program(&prog),
        ast::Node::Statement(stmt) => eval_statement(&stmt),
        ast::Node::Expression(exp) => eval_expression(&exp),
    }
//    Ok(Bytecode{instructions: vec![], constants: vec![]})
    //Err(CompileError{message: "not implemented!".to_string()})
}

fn eval_program(_prog: &ast::Program) -> Result {
    Err(CompileError{message: "not implemented!".to_string()})
}

fn eval_statement(_stmt: &ast::Statement) -> Result {
    Err(CompileError{message: "not implemented!".to_string()})
}

fn eval_expression(_node: &ast::Expression) -> Result {
    Ok(Bytecode{instructions: vec![], constants: vec![]})
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

        if concatted.len() != expected.len() {
            return Err(CompileError{message: format!("instruction lengths not equal\n\texp: {}\n\tgot: {}", concatted.string(), actual.string())})
        }

        let mut pos = 0;

        for (exp, got) in concatted.into_iter().zip(actual) {
            if exp != got {
                return Err(CompileError { message: format!("exp {} but got {} at position {}", exp, got, pos) });
            }
            pos = pos + 1;
        }
        Ok(())
    }

    fn test_constants(expected: &Vec<Object>, actual: Vec<Object>) -> ::std::result::Result<(), CompileError> {
        let mut pos = 0;

        for (exp, got) in expected.into_iter().zip(actual) {
            match (exp, &got) {
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