use byteorder;

use std::io::Cursor;
use self::byteorder::{ByteOrder, BigEndian, WriteBytesExt, ReadBytesExt};

pub type Instructions = Vec<u8>;

pub trait InstructionsFns {
    fn string(&self) -> String;
    fn fmt_instruction(op: &Op, operands: &Vec<usize>) -> String;
}

impl InstructionsFns for Instructions {
    fn string(&self) -> String {
        let mut ret = String::new();
        let mut i = 0;

        while i < self.len() {
            let op:u8 = *self.get(i).unwrap();
            let op = unsafe { ::std::mem::transmute(op) };

            let (operands, read) = read_operands(&op, &self[i+1..]);

            ret.push_str(&format!("{:04} {}\n", i, Self::fmt_instruction(&op, &operands)));
            i = i + 1 + read;
        }

        ret
    }

    fn fmt_instruction(op: &Op, operands: &Vec<usize>) -> String {
        match op.operand_widths().len() {
            1 => format!("{} {}", op.name(), operands.first().unwrap()),
            0 => format!("{}", op.name()),
            _ => panic!("unsuported operand width")
        }
    }
}


#[repr(u8)]
#[derive(Debug, Clone)]
pub enum Op {
    Constant,
    Add,
    Sub,
    Mul,
    Div,
    Pop,
    True,
    False,
    Equal,
    NotEqual,
    GreaterThan,
    Minus,
    Bang,
    JumpNotTruthy,
    Jump,
}

impl Op {
    pub fn name(&self) -> &str {
        match self {
            Op::Constant => "OpConstant",
            Op::Add => "OpAdd",
            Op::Sub => "OpSub",
            Op::Mul => "OpMul",
            Op::Div => "OpDiv",
            Op::Pop => "OpPop",
            Op::True => "OpTrue",
            Op::False => "OpFalse",
            Op::Equal => "OpEqual",
            Op::NotEqual => "OpNotEqual",
            Op::GreaterThan => "OpGreaterThan",
            Op::Minus => "OpMinus",
            Op::Bang => "OpBang",
            Op::JumpNotTruthy => "OpJumpNotTruthy",
            Op::Jump => "OpJump",
        }
    }

    pub fn operand_widths(&self) -> Vec<u8> {
        match self {
            Op::Constant | Op::JumpNotTruthy | Op::Jump => vec![2],
            Op::Add | Op::Sub | Op::Mul |
            Op::Div | Op::Pop | Op::True |
            Op::False | Op::Equal | Op::NotEqual |
            Op::GreaterThan | Op::Minus | Op::Bang => vec![],
        }
    }
}

pub fn make_instruction(op: Op, operands: &Vec<usize>) -> Vec<u8> {
    let mut instruction = Vec::new();
    let widths = op.operand_widths();
    instruction.push(op as u8);

    for (o, width) in operands.into_iter().zip(widths) {
        match width {
            2 => {
                instruction.write_u16::<BigEndian>(*o as u16).unwrap()
            },
            _ => panic!("unsupported operand width {}", width),
        };
    }

    instruction
}

pub fn read_operands(op: &Op, instructions: &[u8]) -> (Vec<usize>, usize) {
    let mut operands = Vec::with_capacity(op.operand_widths().len());
    let mut offset = 0;

    for width in op.operand_widths() {
        match width {
            2 => {
                operands.push(BigEndian::read_u16(&instructions[offset..offset+2]) as usize);
                offset = offset + 2;
            },
            _ => panic!("width not supported for operand")
        }
    }

    (operands, offset)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn make() {
        struct Test {
            op: Op,
            operands: Vec<usize>,
            expected: Vec<u8>,
        }

        let tests = vec![
            Test{op: Op::Constant, operands: vec![65534], expected: vec![Op::Constant as u8, 255, 254]},
            Test{op: Op::Add, operands: vec![], expected: vec![Op::Add as u8]},
        ];

        for t in tests {
            let instruction = make_instruction(t.op, &t.operands);
            assert_eq!(t.expected, instruction)
        }
    }

    #[test]
    fn instructions_string() {
        let instructions = vec![
            make_instruction(Op::Add, &vec![]),
            make_instruction(Op::Constant, &vec![2]),
            make_instruction(Op::Constant, &vec![65535]),
        ];
        let expected = "0000 OpAdd\n0001 OpConstant 2\n0004 OpConstant 65535\n";
        let concatted = instructions.into_iter().flatten().collect::<Instructions>();

        assert_eq!(expected, concatted.string())
    }

    #[test]
    fn test_read_operands() {
        struct Test {
            op: Op,
            operands: Vec<usize>,
            bytes_read: usize,
        }

        let tests = vec![
            Test{op: Op::Constant, operands: vec![65535], bytes_read: 2},
        ];

        for t in tests {
            let instruction = make_instruction(t.op, &t.operands);

            let (operands_read, n) = read_operands(&Op::Constant, &instruction[1..]);
            assert_eq!(n, t.bytes_read);
            assert_eq!(operands_read, t.operands);
        }
    }
}