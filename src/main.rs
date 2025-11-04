mod bootstrap;
mod io;
mod observables;
mod parser;
mod spectroscopy;
mod statistics;
mod wilsonflow;
use parser::run;

fn main() {
    run();
}
