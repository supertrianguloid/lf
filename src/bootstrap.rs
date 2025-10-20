use crate::parser::BinBootstrapArgs;
use crate::statistics::{bin, mean, Histogram};
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Serialize, Deserialize)]
pub enum BootstrapResult {
    SingleBootstrap {
        n_boot: usize,
        replicas: Vec<f64>,
        central_val: f64,
        z: f64,
        a: f64,
        median: f64,
        mean: f64,
        ci_68: (f64, f64),
        ci_95: (f64, f64),
        ci_99: (f64, f64),
        failed_samples: usize,
        histogram: Histogram,
    },
    DoubleBootstrap(Vec<Vec<f64>>),
}
impl BootstrapResult {
    pub fn print(&self) {
        println!("{}", serde_json::to_string(&self).unwrap());
    }
    pub fn get_single_bootstrap_result(self) -> Vec<f64> {
        match self {
            BootstrapResult::SingleBootstrap {
                n_boot: _,
                replicas: v,
                central_val: _,
                z: _,
                a: _,
                ci_68: _,
                ci_95: _,
                ci_99: _,
                median: _,
                mean: _,
                histogram: _,
                failed_samples: _,
            } => v,
            BootstrapResult::DoubleBootstrap(_) => unimplemented!(),
        }
    }
}
// #[inline(always)]
pub fn get_samples(length: usize, binsize: usize) -> Vec<usize> {
    let length_new = length / binsize;
    let mut rng = rand::rng();
    let samples: Vec<_> = Uniform::try_from(0..length)
        .unwrap()
        .sample_iter(&mut rng)
        .take(length_new)
        .collect();
    samples
}

// #[inline(always)]
pub fn get_subsample(sample: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(sample.len());
    let mut rng = rand::rng();
    for _ in 0..sample.len() {
        let index = rng.random_range(..sample.len());
        result.push(sample[index]);
    }
    result
}
pub fn bootstrap<T>(func: T, length: usize, boot_args: &BinBootstrapArgs) -> BootstrapResult
where
    T: Fn(Vec<usize>) -> Option<f64> + Sync + Send,
{
    fn drop_nones(results: Vec<Option<f64>>) -> Vec<f64> {
        let mut results_g = vec![];
        for result in results {
            match result {
                None => {}
                Some(val) => results_g.push(val),
            };
        }
        results_g
    }

    if let Some(n_boot_double) = boot_args.n_boot_double {
        BootstrapResult::DoubleBootstrap(
            (0..n_boot_double)
                .into_par_iter()
                .map(|_| {
                    let sample = get_samples(length, boot_args.binwidth);
                    drop_nones(
                        (0..boot_args.n_boot)
                            .map(|_| func(get_subsample(&sample)))
                            .collect::<Vec<Option<f64>>>(),
                    )
                })
                .collect::<Vec<Vec<f64>>>(),
        )
    } else {
        let norm = Normal::standard();
        let mut replicas = drop_nones(
            (0..boot_args.n_boot)
                .into_par_iter()
                .map(|_| func(get_samples(length, boot_args.binwidth)))
                .collect(),
        );
        replicas.par_sort_unstable_by(f64::total_cmp);
        let central_val =
            func((0..length).collect()).expect("Should be able to evaluate the central value!");
        let mut prop = 0.0;
        for replica in &replicas {
            if *replica < central_val {
                prop += 1.0
            } else if *replica == central_val {
                prop += 0.5
            }
        }
        let z = norm.inverse_cdf(prop / replicas.len() as f64);

        // let jack = jackknife_samples(func, length);
        // let jack_avg = jack.iter().sum::<f64>() / (length as f64);
        // let mut num = 0.0;
        // let mut denom = 0.0;

        // for i in 0..length {
        // num += (jack_avg - jack[i]).powi(3);
        // denom += (jack_avg - jack[i]).powi(2);
        // }

        // let a = num / (6.0 * denom.powf(3.0 / 2.0));
        let a = 0.0;

        BootstrapResult::SingleBootstrap {
            ci_68: confidence_interval(&replicas, z, a, 1.0 - 0.682689492137086),
            ci_95: confidence_interval(&replicas, z, a, 1.0 - 0.954499736103642),
            ci_99: confidence_interval(&replicas, z, a, 1.0 - 0.997300203936740),
            n_boot: boot_args.n_boot,
            median: replicas[replicas.len() / 2],
            mean: mean(&replicas),
            histogram: bin(&replicas, boot_args.n_bins_histogram),
            failed_samples: boot_args.n_boot - replicas.len(),
            replicas: replicas,
            central_val: central_val,
            z: z,
            a: a,
        }
    }
}
pub fn jackknife_samples<T>(func: T, length: usize) -> Vec<f64>
where
    T: Fn(Vec<usize>) -> Option<f64> + Sync + Send,
{
    let mut samples = vec![];
    let indices: Vec<usize> = (0..length).collect();
    for i in 0..length {
        let mut cur_indices = indices.clone();
        cur_indices.remove(i);
        samples.push(func(cur_indices).unwrap());
    }
    samples
}

pub fn confidence_interval(replicas: &[f64], z: f64, a: f64, alpha: f64) -> (f64, f64) {
    let norm = Normal::standard();
    let z_a1 = norm.inverse_cdf(alpha / 2.0);
    let z_a2 = norm.inverse_cdf(1.0 - alpha / 2.0);
    let lower = norm.cdf(z + (z + z_a1) / (1.0 - a * (z + z_a1)));
    let upper = norm.cdf(z + (z + z_a2) / (1.0 - a * (z + z_a2)));
    return (
        replicas[((replicas.len() as f64) * lower).round() as usize],
        replicas[((replicas.len() as f64) * upper).round() as usize],
    );
}

#[cfg(test)]
#[test]
fn test_bootstrap_binning_samples() {
    assert_eq!(get_samples(100, 2).len(), 50);
}
#[test]
fn test_bootstrap_no_binning_samples() {
    assert_eq!(get_samples(100, 1).len(), 100);
}
#[test]
fn test_jackknife() {
    assert_eq!(
        jackknife_samples(
            |x| {
                let vals = vec![1.0, 2.0, 3.0];
                let mut acc = 0.0;
                for i in &x {
                    acc += vals[*i]
                }
                Some(acc / (x.len() as f64))
            },
            3
        ),
        vec![2.5, 2.0, 1.5]
    );
}
