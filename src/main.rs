mod io;
mod statistics;
use io::{
    load_channel_from_file_folded, load_global_t_from_file, load_wf_observable_from_file,
    Observable, WfObservable,
};
use serde::{Deserialize, Serialize};
use spectroscopy::effective_mass;
use statistics::{mean, standard_deviation};
mod spectroscopy;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{stdin, stdout},
};

#[derive(Parser, Debug)]
pub struct App {
    #[clap(subcommand)]
    command: Command,
}
#[derive(Subcommand, Debug)]
enum Command {
    /// Calculate the effective mass in a given channel
    ComputeEffectiveMass {
        #[clap(flatten)]
        args: ComputeEffectiveMassArgs,
    },
    /// Given a CSV generated from compute-effective-mass, fit a constant to it
    FitEffectiveMass {
        #[clap(flatten)]
        args: FitEffectiveMassArgs,
    },
}

#[derive(Parser, Debug)]
struct ComputeEffectiveMassArgs {
    hmc_filename: String,
    #[arg(short, long, value_name = "CHANNEL")]
    channel: String,
    #[arg(short, long, value_name = "NUM_CONFS", default_value_t = 0)]
    thermalisation: usize,
    #[arg(short, long, value_name = "NUM_SAMPLES", default_value_t = 1000)]
    n_boot: u32,
    #[arg(short, long, value_name = "BIN_WIDTH", default_value_t = 10)]
    binwidth: usize,
    #[arg(short, long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
    #[arg(short, long, value_name = "WILSON_FLOW_FILE")]
    wilson_flow_filename: Option<String>,
    #[arg(long, value_name = "WILSON_FLOW_NUM_MEASUREMENTS")]
    wilson_flow_nmeas: Option<usize>,
}
#[derive(Parser, Debug)]
struct FitEffectiveMassArgs {
    csv_filename: String,
    t1: usize,
    t2: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct EffectiveMassRow {
    #[serde(rename = "Tau")]
    tau: usize,
    #[serde(rename = "Effective Mass")]
    mass: f64,
    #[serde(rename = "Error")]
    error: f64,
    #[serde(rename = "Failed Samples (%)")]
    failures: f64,
}
#[derive(Debug, Serialize)]
struct EffectiveMassFit {
    #[serde(rename = "Effective Mass Fit")]
    mass: f64,
    #[serde(rename = "Error")]
    error: f64,
}

fn main() {
    let app = App::parse();
    match app.command {
        Command::ComputeEffectiveMass { args } => compute_effective_mass_command(args),
        Command::FitEffectiveMass { args } => fit_effective_mass_command(args),
    }
}
fn fit_effective_mass_command(args: FitEffectiveMassArgs) {
    let mut tau = vec![];
    let mut mass = vec![];
    let mut error = vec![];
    let mut rdr = csv::Reader::from_reader(File::open(args.csv_filename).unwrap());
    for result in rdr.deserialize() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let record: EffectiveMassRow = result.unwrap();
        tau.push(record.tau);
        mass.push(record.mass);
        error.push(record.error);
    }
    let offset = tau.iter().position(|&x| x == args.t1).unwrap();
    let index = offset..(offset + args.t2 - args.t1 + 1);
    let fit = statistics::weighted_mean(&mass[index.clone()], &error[index]);
    let mut wtr = csv::Writer::from_writer(stdout());
    wtr.serialize(EffectiveMassFit {
        mass: fit.0,
        error: fit.1,
    })
    .unwrap();
    wtr.flush().unwrap();
}

fn compute_effective_mass_command(args: ComputeEffectiveMassArgs) {
    let channel = load_channel_from_file_folded(&args.hmc_filename, &args.channel)
        .thermalise(args.thermalisation);
    //if let Some(wf_filename) = args.wilson_flow_filename {
    //load_wf_observables_from_file(&wf_filename, WfObservable::T);
    //}

    let global_t = load_global_t_from_file(&args.hmc_filename);

    let mut solve_failures = vec![];
    let mut effmass_mean = vec![];
    let mut effmass_error = vec![];
    for tau in 1..=args.effective_mass_t_max {
        let results: Vec<Result<f64, roots::SearchError>> = (0..args.n_boot)
            .into_par_iter()
            .map(|_| {
                let (mu, _) = channel.get_subsample_mean_stderr(args.binwidth);
                effective_mass(&mu, global_t, tau)
            })
            .collect();
        let mut effmass_inner = Vec::with_capacity(args.n_boot as usize);
        let mut nfailures = 0;
        for result in results {
            match result {
                Ok(val) => effmass_inner.push(val),
                Err(_) => nfailures += 1,
            }
        }
        solve_failures.push(nfailures);
        effmass_mean.push(mean(&effmass_inner));
        effmass_error.push(standard_deviation(&effmass_inner, true));
    }
    let mut wtr = csv::Writer::from_writer(stdout());
    for tau in 2..=args.effective_mass_t_max {
        wtr.serialize(EffectiveMassRow {
            tau,
            mass: effmass_mean[tau - 1],
            error: effmass_error[tau - 1],
            failures: solve_failures[tau - 1] as f64 * 100.0 / args.n_boot as f64,
        })
        .unwrap();
        wtr.flush().unwrap();
        /*         println!(
            "{:0>2},{},{},{}",
            tau,
            effmass_mean[tau - 1],
            effmass_error[tau - 1],
            solve_failures[tau - 1] as f64 * 100.0 / args.n_boot as f64
        ); */
    }
}
