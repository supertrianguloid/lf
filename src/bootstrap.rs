use crate::parser::BinBootstrapArgs;
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum BootstrapResult {
    SingleBootstrap {
        replicas: Vec<f64>,
        central_val: f64,
        z: f64,
        a: f64,
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
                replicas: v,
                central_val: _,
                z: _,
                a: _,
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
        let replicas = drop_nones(
            (0..boot_args.n_boot)
                .into_par_iter()
                .map(|_| func(get_samples(length, boot_args.binwidth)))
                .collect(),
        );
        let central_val =
            func((0..length).collect()).expect("Should be able to evaluate the central value!");
        BootstrapResult::SingleBootstrap {
            replicas: replicas,
            central_val: central_val,
            z: 0.0,
            a: 0.0,
        }
    }
}
pub fn jackknife_samples<T, F>(data: &[T], stat: F) -> Vec<f64>
where
    T: Clone,
    F: Fn(&[T]) -> f64,
{
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    let mut buf = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n {
        buf.clear();
        buf.extend_from_slice(&data[..i]);
        buf.extend_from_slice(&data[i + 1..]);
        out.push(stat(&buf));
    }
    out
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
        jackknife_samples(&vec![1.0, 2.0, 3.0], |x| x.iter().sum()),
        vec![100.0]
    );
}
