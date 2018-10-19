extern crate monkey;

use monkey::repl;
use std::io;

fn main() -> io::Result<()> {
    println!("Welcome to the Monkey REPL!");
    let input = io::stdin();
    let output = io::stdout();
    let result = repl::start(input.lock(), output.lock());
    result
}
