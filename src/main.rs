mod bootstrap;
mod io;
mod observables;
mod parser;
mod spectroscopy;
mod statistics;
mod wilsonflow;
use parser::parser;

fn main() {
    parser();
}
