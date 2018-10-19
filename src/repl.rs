use std::io;
use std::cell::RefCell;
use std::rc::Rc;
use parser;
use evaluator;
use object::Environment;

pub fn start<R: io::BufRead, W: io::Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    #![allow(warnings)]
    let mut env = Rc::new(RefCell::new(Environment::new()));
    loop {
        writer.write(b"> ");
        writer.flush();
        let mut line = String::new();
        reader.read_line(&mut line)?;

        match parser::parse(&line) {
            Ok(node) => {
                match evaluator::eval(&node, Rc::clone(&env)) {
                    Ok(n) => write!(writer, "{}\n", n.inspect()),
                    Err(e) => write!(writer, "error: {}\n", e.message),
                };
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
