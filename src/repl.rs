use std::io;
use crate::parser;
use crate::compiler::compile;
use crate::vm;

pub fn start<R: io::BufRead, W: io::Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    #![allow(warnings)]
    loop {
        writer.write(b"> ");
        writer.flush();
        let mut line = String::new();
        reader.read_line(&mut line)?;

        match parser::parse(&line) {
            Ok(node) => {
                match compile(node) {
                    Ok(bytecode) => {
                        let mut vm = vm::VM::new(bytecode.constants, bytecode.instructions);
                        vm.run();
                        write!(writer, "{:?}\n", vm.last_popped_stack_elem().unwrap().inspect());
                    },
                    Err(e) => {
                        write!(writer, "error: {}\n", e.message);
                    },
                }
            },
            Err(errors) => {
                for err in errors {
                    write!(writer, "parse errors:\n{}\n", err.to_string());
                };
            },
        }
    }
    Ok(())
}
