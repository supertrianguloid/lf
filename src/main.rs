mod io;
mod statistics;
use io::{
    load_channel_from_file_folded, load_global_t_from_file, load_wf_observable_from_file,
    Observable, WfObservable,
};
use spectroscopy::effective_mass;
use statistics::{mean, standard_deviation};
mod spectroscopy;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
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
    #[arg(short, long, value_name = "WILSON_FLOW_NUM_MEASUREMENTS")]
    wilson_flow_nmeas: Option<usize>,
}

fn main() {
    let args = Args::parse();
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
        let mut effmass_inner = vec![];
        let mut nfailures = 0;
        for _ in 0..args.n_boot {
            let (mu, _) = channel.get_subsample_mean_stderr(args.binwidth);
            match effective_mass(&mu, global_t, tau) {
                Ok(val) => effmass_inner.push(val),
                Err(_) => nfailures += 1,
            }
        }
        solve_failures.push(nfailures);
        effmass_mean.push(mean(&effmass_inner));
        effmass_error.push(standard_deviation(&effmass_inner, true));
    }
    println!("Tau,Effective Mass,Error,Failed Samples (%)");
    for tau in 2..=args.effective_mass_t_max {
        println!(
            "{:0>2},{},{},{}",
            tau,
            effmass_mean[tau - 1],
            effmass_error[tau - 1],
            solve_failures[tau - 1] as f64 * 100.0 / args.n_boot as f64
        );
    }
}
