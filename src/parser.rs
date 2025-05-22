use crate::bootstrap::{bootstrap, BootstrapResult};
use crate::observables::{Measurement, ObservableCalculation};
use crate::spectroscopy::{effective_mass, effective_mass_all_t};
use crate::statistics::{bin, mean, standard_deviation, weighted_mean};
use crate::wilsonflow::{calculate_w0_from_samples, WilsonFlowCalculation};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::generate;
use clap_complete_nushell::Nushell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::stdout};

#[derive(Parser, Debug)]
#[clap(
    name = "Reshotka",
    version = "0.0.1",
    author = "Laurence Sebastian Bowes",
    about = "A tool for SU(2) analysis"
)]
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
    /// Given an effective mass generated from `compute-effective-mass`, fit a constant to it
    FitEffectiveMass {
        #[clap(flatten)]
        args: FitEffectiveMassArgs,
    },
    BootstrapFits {
        #[clap(flatten)]
        args: BootstrapFitsArgs,
    },
    BootstrapFitsRatio {
        #[clap(flatten)]
        args: BootstrapFitsRatioArgs,
    },
    CalculateW0 {
        #[clap(flatten)]
        args: CalculateW0Args,
    },
    Histogram {
        #[clap(flatten)]
        args: HistogramArgs,
    },
    GenerateCompletions {},
}

#[derive(Parser, Debug)]
pub struct HMCArgs {
    pub filename: String,
    #[arg(short, long, value_name = "THERMALISATION", default_value_t = 0)]
    pub thermalisation: usize,
}

#[derive(Parser, Debug)]
#[group(requires = "wf_filename")]
pub struct WFArgs {
    #[arg(long, value_name = "WILSON_FLOW_FILE", required = false)]
    pub wf_filename: String,
    #[arg(long, value_name = "W_THERMALISATION", default_value_t = 0)]
    pub wf_thermalisation: usize,
    #[arg(long, value_name = "W_REFERENCE", default_value_t = 1.0)]
    pub w_ref: f64,
}

#[derive(Parser, Debug)]
pub struct BinBootstrapArgs {
    #[arg(short, long, value_name = "BOOTSTRAP_SAMPLES", default_value_t = 1000)]
    pub n_boot: usize,
    #[arg(short, long, value_name = "BIN_WIDTH", default_value_t = 1)]
    pub binwidth: usize,
    #[arg(long, value_name = "DOUBLE_BOOTSTRAP_SAMPLES")]
    pub n_boot_double: Option<u32>,
}

#[derive(Parser, Debug)]
struct ComputeEffectiveMassArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(short, long, value_name = "CHANNEL")]
    channel: String,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
}

#[derive(Parser, Debug)]
struct FitEffectiveMassArgs {
    json_filename: String,
    t1: usize,
    t2: usize,
}

#[derive(Parser, Debug)]
struct HistogramArgs {
    json_filename: String,
    nbins: usize,
}
#[derive(Parser, Debug)]
struct CalculateW0Args {
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[clap(flatten)]
    wf: WFArgs,
}

#[derive(Parser, Debug)]
struct BootstrapFitsArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(short, long, value_name = "CHANNEL")]
    channel: String,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
    #[clap(flatten)]
    wf: Option<WFArgs>,
}

#[derive(Parser, Debug)]
struct BootstrapFitsRatioArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(long, value_name = "NUMERATOR_CHANNEL")]
    numerator_channel: String,
    #[arg(long, value_name = "DENOMINATOR_CHANNEL")]
    denominator_channel: String,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
    #[arg(long, value_name = "NUMERATOR_EFFECTIVE_MASS_T_MAX")]
    numerator_effective_mass_t_max: usize,
    #[arg(long, value_name = "NUMERATOR_EFFECTIVE_MASS_T_MIN")]
    numerator_effective_mass_t_min: usize,
    #[arg(long, value_name = "DENOMINATOR_EFFECTIVE_MASS_T_MAX")]
    denominator_effective_mass_t_max: usize,
    #[arg(long, value_name = "DENOMINATOR_EFFECTIVE_MASS_T_MIN")]
    denominator_effective_mass_t_min: usize,
}

#[derive(Parser, Debug)]
struct BootstrapErrorArgs {
    json_filename: String,
    #[arg(short, long, value_name = "BOOTSTRAP_SAMPLES", default_value_t = 1000)]
    n_boot: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct EffectiveMass {
    #[serde(rename = "Tau")]
    tau: Vec<usize>,
    #[serde(rename = "Effective Mass")]
    mass: Vec<f64>,
    #[serde(rename = "Error")]
    error: Vec<f64>,
    #[serde(rename = "Failed Samples (%)")]
    failures: Vec<usize>,
}

fn fit_effective_mass_command(args: FitEffectiveMassArgs) {
    let EffectiveMass {
        tau, mass, error, ..
    } = serde_json::from_reader(File::open(args.json_filename).unwrap()).unwrap();
    let offset = tau.iter().position(|&x| x == args.t1).unwrap();
    let index = offset..(offset + args.t2 - args.t1 + 1);
    let fit = weighted_mean(&mass[index.clone()], &error[index]);
    println!("{}", serde_json::to_string(&fit).unwrap());
}

fn compute_effective_mass_command(args: ComputeEffectiveMassArgs) {
    let channel = ObservableCalculation::load(&args.hmc, args.channel);

    let mut solve_failures = vec![];
    let mut effmass_mean = vec![];
    let mut effmass_error = vec![];
    assert_eq!(channel.global_t, (channel.obs.each_len - 1) * 2);
    for tau in args.effective_mass_t_min..=args.effective_mass_t_max {
        let results: Vec<Result<f64, roots::SearchError>> = (0..args.boot.n_boot)
            .into_par_iter()
            .map(|_| {
                let Measurement {
                    values: mu,
                    errors: _,
                } = channel.obs.get_subsample_mean_stderr(args.boot.binwidth);
                effective_mass(&mu, channel.global_t, tau, args.solver_precision)
            })
            .collect();
        let mut effmass_inner = Vec::with_capacity(args.boot.n_boot as usize);
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
    println!(
        "{}",
        serde_json::to_string(&EffectiveMass {
            tau: (args.effective_mass_t_min..=args.effective_mass_t_max).collect(),
            mass: effmass_mean,
            error: effmass_error,
            failures: solve_failures,
        })
        .unwrap()
    )
}

fn bootstrap_fits_command(args: BootstrapFitsArgs) {
    let channel = ObservableCalculation::load(&args.hmc, args.channel);

    let wf = if let Some(wfargs) = args.wf {
        Some(WilsonFlowCalculation::load(wfargs))
    } else {
        None
    };
    let func = |samples: Vec<usize>| {
        let mu = &channel
            .obs
            .get_subsample_mean_stderr_from_samples(&samples)
            .values;
        let masses = effective_mass_all_t(
            &mu,
            channel.global_t,
            args.effective_mass_t_min,
            args.effective_mass_t_max,
            args.solver_precision,
        )?;
        let factor = match &wf {
            None => 1.0,
            Some(wf) => calculate_w0_from_samples(&wf.data, &samples, wf.w_ref)?,
        };
        Some(mean(&masses) * factor)
    };
    let results = bootstrap(func, channel.obs.nconfs, args.boot);
    results.print();
}
fn bootstrap_fits_ratio_command(args: BootstrapFitsRatioArgs) {
    let numerator_channel = ObservableCalculation::load(&args.hmc, args.numerator_channel);
    let denominator_channel = ObservableCalculation::load(&args.hmc, args.denominator_channel);
    let func = |samples: Vec<usize>| {
        let num_mu = numerator_channel
            .obs
            .get_subsample_mean_stderr_from_samples(&samples)
            .values;
        let num_masses = effective_mass_all_t(
            &num_mu,
            numerator_channel.global_t,
            args.numerator_effective_mass_t_min,
            args.numerator_effective_mass_t_max,
            args.solver_precision,
        )?;

        let denom_mu = denominator_channel
            .obs
            .get_subsample_mean_stderr_from_samples(&samples)
            .values;
        let denom_masses = effective_mass_all_t(
            &denom_mu,
            denominator_channel.global_t,
            args.denominator_effective_mass_t_min,
            args.denominator_effective_mass_t_max,
            args.solver_precision,
        )?;

        Some(mean(&num_masses) / mean(&denom_masses))
    };
    let results = bootstrap(func, numerator_channel.obs.nconfs, args.boot);

    results.print();
}
fn calculate_w0_command(args: CalculateW0Args) {
    let wf = WilsonFlowCalculation::load(args.wf);
    let func = |samples: Vec<usize>| calculate_w0_from_samples(&wf.data, &samples, wf.w_ref);
    let results = bootstrap(func, wf.data.t2_esym.nconfs, args.boot);
    results.print();
}

fn histogram_command(args: HistogramArgs) {
    if let BootstrapResult::SingleBootstrap(mut sample) =
        serde_json::from_reader(File::open(args.json_filename).unwrap()).unwrap()
    {
        sample.sort_by(f64::total_cmp);
        let hist = bin(&sample, args.nbins);
        println!("{}", serde_json::to_string(&hist).unwrap());
    }
}

pub fn parser() {
    let app = App::parse();
    match app.command {
        Command::ComputeEffectiveMass { args } => compute_effective_mass_command(args),
        Command::FitEffectiveMass { args } => fit_effective_mass_command(args),
        Command::BootstrapFits { args } => bootstrap_fits_command(args),
        Command::BootstrapFitsRatio { args } => bootstrap_fits_ratio_command(args),
        Command::CalculateW0 { args } => calculate_w0_command(args),
        Command::Histogram { args } => histogram_command(args),
        Command::GenerateCompletions {} => {
            generate(Nushell, &mut App::command(), "reshotka", &mut stdout())
        }
    }
}
