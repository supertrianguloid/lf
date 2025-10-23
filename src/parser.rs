use crate::bootstrap::{bootstrap, get_samples, BootstrapResult};
use crate::io::load_plaquette_from_file;
use crate::observables::{Measurement, Observable, ObservableCalculation};
use crate::spectroscopy::{effective_mass, effective_mass_all_t, effective_pcac, fit_cosh};
use crate::statistics::{bin, mean, median, standard_deviation, weighted_mean};
use crate::wilsonflow::{calculate_w0_from_samples, extract_tc, WilsonFlowCalculation};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::generate;
use clap_complete_nushell::Nushell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fs::read_to_string, io::stdout};

#[derive(Parser, Debug)]
#[clap(
    name = "lf",
    version = "0.0.1",
    author = "Laurence Sebastian Bowes",
    about = "A tool for fitting lattice data."
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
    BootstrapCorrelatorFits {
        #[clap(flatten)]
        args: BootstrapCorrelatorFitsArgs,
    },
    BootstrapFitsRatio {
        #[clap(flatten)]
        args: BootstrapFitsRatioArgs,
    },
    CalculateW0 {
        #[clap(flatten)]
        args: CalculateW0Args,
    },
    ExtractTC {
        #[clap(flatten)]
        args: ExtractTCArgs,
    },
    /// Extract the plaquette. Must be run on the WF data.
    Plaquette {
        #[clap(flatten)]
        args: PlaquetteArgs,
    },
    ComputePCACMass {
        #[clap(flatten)]
        args: ComputePCACMassArgs,
    },
    ComputePCACMassFit {
        #[clap(flatten)]
        args: ComputePCACMassFitArgs,
    },
    Median {
        #[clap(flatten)]
        args: MedianArgs,
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
pub struct PlaquetteArgs {
    pub filename: String,
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
    #[arg(long, value_name = "HISTOGRAM_BINS", default_value_t = 1000)]
    pub n_bins_histogram: usize,
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
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
}

#[derive(Parser, Debug)]
struct FitEffectiveMassArgs {
    json_filename: String,
    t1: usize,
    t2: usize,
}

#[derive(Parser, Debug)]
struct CalculateW0Args {
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[clap(flatten)]
    wf: WFArgs,
}
#[derive(Parser, Debug)]
struct ExtractTCArgs {
    #[arg(long, value_name = "T_REFERENCE", default_value_t = 0.0)]
    pub t_ref: f64,
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
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
    #[clap(flatten)]
    wf: Option<WFArgs>,
}

#[derive(Parser, Debug)]
struct BootstrapCorrelatorFitsArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(short, long, value_name = "CHANNEL")]
    channel: String,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
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
    #[arg(long, value_name = "NUMERATOR_EFFECTIVE_MASS_T_MIN")]
    numerator_effective_mass_t_min: usize,
    #[arg(long, value_name = "NUMERATOR_EFFECTIVE_MASS_T_MAX")]
    numerator_effective_mass_t_max: usize,
    #[arg(long, value_name = "DENOMINATOR_EFFECTIVE_MASS_T_MIN")]
    denominator_effective_mass_t_min: usize,
    #[arg(long, value_name = "DENOMINATOR_EFFECTIVE_MASS_T_MAX")]
    denominator_effective_mass_t_max: usize,
}

#[derive(Parser, Debug)]
struct ComputePCACMassArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
}
#[derive(Parser, Debug)]
struct MedianArgs {
    json_filename: String,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(short, long, value_name = "THERMALISATION", default_value_t = 0)]
    thermalisation: usize,
}

#[derive(Parser, Debug)]
struct ComputePCACMassFitArgs {
    #[clap(flatten)]
    hmc: HMCArgs,
    #[clap(flatten)]
    boot: BinBootstrapArgs,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MIN")]
    effective_mass_t_min: usize,
    #[arg(long, value_name = "EFFECTIVE_MASS_T_MAX")]
    effective_mass_t_max: usize,
    #[arg(short, long, value_name = "SOLVER_PRECISION", default_value_t = 1e-15)]
    solver_precision: f64,
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
    failures: Vec<f64>,
}
#[derive(Deserialize)]
struct Data {
    data: Vec<f64>,
}

fn fit_effective_mass_command(args: FitEffectiveMassArgs) {
    let EffectiveMass {
        tau, mass, error, ..
    } = serde_json::from_str(&read_to_string(args.json_filename).unwrap()).unwrap();
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
        let mut effmass_inner = Vec::with_capacity(args.boot.n_boot);
        let mut nfailures = 0;
        for result in results {
            match result {
                Ok(val) => effmass_inner.push(val),
                Err(_) => nfailures += 1,
            }
        }
        solve_failures.push(100.0 * (nfailures as f64) / (args.boot.n_boot as f64));
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
fn bootstrap_correlator_fits_command(args: BootstrapCorrelatorFitsArgs) {
    let channel = ObservableCalculation::load(&args.hmc, args.channel);
    let func = |samples: Vec<usize>| {
        let corr = &channel.obs.get_subsample_mean_stderr_from_samples(&samples);
        let masses = fit_cosh(
            corr,
            channel.global_t,
            args.effective_mass_t_min,
            args.effective_mass_t_max,
        );
        Some(masses.mass)
    };
    bootstrap(func, channel.obs.nconfs, &args.boot).print();
}
fn bootstrap_fits_command(args: BootstrapFitsArgs) {
    let channel = ObservableCalculation::load(&args.hmc, args.channel);

    let wf = if let Some(wfargs) = args.wf {
        let w = Some(WilsonFlowCalculation::load(wfargs));
        assert_eq!(channel.obs.nconfs, w.as_ref().unwrap().data.tc.nconfs);
        w
    } else {
        None
    };
    let func = |samples: Vec<usize>| {
        let mu = &channel
            .obs
            .get_subsample_mean_stderr_from_samples(&samples)
            .values;
        let masses = effective_mass_all_t(
            mu,
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
    let results = bootstrap(func, channel.obs.nconfs, &args.boot);
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
    let results = bootstrap(func, numerator_channel.obs.nconfs, &args.boot);

    results.print();
}
fn calculate_w0_command(args: CalculateW0Args) {
    let wf = WilsonFlowCalculation::load(args.wf);
    let func = |samples: Vec<usize>| calculate_w0_from_samples(&wf.data, &samples, wf.w_ref);
    let results = bootstrap(func, wf.data.t2_esym.nconfs, &args.boot);
    results.print();
}

fn extract_tc_command(args: ExtractTCArgs) {
    let wf = WilsonFlowCalculation::load(args.wf);
    println!(
        "{}",
        serde_json::to_string(&extract_tc(wf.data, args.t_ref).unwrap()).unwrap()
    )
}
fn compute_effective_pcac_mass_command(args: ComputePCACMassArgs) {
    let f_ap = ObservableCalculation::load(&args.hmc, String::from("g5_g0g5_re"));
    let f_ps = ObservableCalculation::load(&args.hmc, String::from("g5"));

    let mut central_val = vec![];
    let mut errors = vec![];

    for t in 0..(f_ap.obs.each_len - 2) {
        let func = |samples: Vec<usize>| {
            Some(effective_pcac(
                &f_ap
                    .obs
                    .get_subsample_mean_stderr_from_samples(&samples)
                    .values,
                &f_ps
                    .obs
                    .get_subsample_mean_stderr_from_samples(&samples)
                    .values,
                &effective_mass_all_t(
                    &f_ps
                        .obs
                        .get_subsample_mean_stderr_from_samples(&samples)
                        .values,
                    f_ps.global_t,
                    1,
                    f_ap.global_t / 2,
                    args.solver_precision,
                )?,
                t,
            ))
        };
        central_val.push(func(get_samples(f_ap.obs.nconfs, args.boot.binwidth)).unwrap());
        errors.push(standard_deviation(
            &bootstrap(func, f_ap.obs.nconfs, &args.boot).get_single_bootstrap_result(),
            true,
        ));
    }

    // let func = |samples: Vec<usize>| calculate_w0_from_samples(&wf.data, &samples, wf.w_ref);

    // let results = bootstrap(func, wf.data.t2_esym.nconfs, args.boot);

    println!(
        "{}",
        serde_json::to_string(&EffectiveMass {
            tau: (0..(f_ap.obs.each_len - 2)).collect(),
            mass: central_val,
            error: errors,
            failures: std::iter::repeat(0.0).take(f_ap.obs.each_len - 2).collect(),
        })
        .unwrap()
    )
}
fn bootstrap_pcac_fit_command(args: ComputePCACMassFitArgs) {
    let f_ap = ObservableCalculation::load(&args.hmc, String::from("g5_g0g5_re"));
    let f_ps = ObservableCalculation::load(&args.hmc, String::from("g5"));

    let func = |samples: Vec<usize>| {
        let mut mass = vec![];
        for t in args.effective_mass_t_min..=args.effective_mass_t_max {
            let m_ps_eff = effective_mass_all_t(
                &f_ps
                    .obs
                    .get_subsample_mean_stderr_from_samples(&samples)
                    .values,
                f_ps.global_t,
                1,
                f_ap.global_t / 2,
                args.solver_precision,
            )?;

            mass.push(effective_pcac(
                &f_ap
                    .obs
                    .get_subsample_mean_stderr_from_samples(&samples)
                    .values,
                &f_ps
                    .obs
                    .get_subsample_mean_stderr_from_samples(&samples)
                    .values,
                &m_ps_eff,
                t,
            ));
        }
        Some(mean(&mass))
    };

    let results = bootstrap(func, f_ap.obs.nconfs, &args.boot);

    results.print()
}

fn median_command(args: MedianArgs) {
    if let Data { data: mut data } =
        serde_json::from_str(&read_to_string(args.json_filename).unwrap()).unwrap()
    {
        let data = data.split_off(args.thermalisation);
        let func = |samples: Vec<usize>| {
            let mut smp = vec![];
            for sample in samples {
                smp.push(data[sample]);
            }
            return Some(median(&smp));
        };
        bootstrap(func, data.len(), &args.boot).print()
    }
}

fn plaquette_command(args: PlaquetteArgs) {
    let plaq = load_plaquette_from_file(&args.filename);
    println!("{}", serde_json::to_string(&plaq).unwrap());
}

pub fn parser() {
    let app = App::parse();
    match app.command {
        Command::ComputeEffectiveMass { args } => compute_effective_mass_command(args),
        Command::FitEffectiveMass { args } => fit_effective_mass_command(args),
        Command::BootstrapFits { args } => bootstrap_fits_command(args),
        Command::BootstrapCorrelatorFits { args } => bootstrap_correlator_fits_command(args),
        Command::BootstrapFitsRatio { args } => bootstrap_fits_ratio_command(args),
        Command::CalculateW0 { args } => calculate_w0_command(args),
        Command::ExtractTC { args } => extract_tc_command(args),
        Command::Median { args } => median_command(args),
        Command::Plaquette { args } => plaquette_command(args),
        Command::ComputePCACMass { args } => compute_effective_pcac_mass_command(args),
        Command::ComputePCACMassFit { args } => bootstrap_pcac_fit_command(args),
        Command::GenerateCompletions {} => {
            generate(Nushell, &mut App::command(), "reshotka", &mut stdout())
        }
    }
}
