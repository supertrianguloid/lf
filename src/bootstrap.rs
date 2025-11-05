use crate::parser::BootstrapArgs;
use crate::statistics::{Histogram, bin, standard_deviation};
// use nalgebra::DVector;
use rand::Rng;
use rand_distr::multi::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// use statrs::distribution::{ContinuousCDF, Normal};
use std::iter;

// #[derive(Debug, Serialize, Deserialize)]
// pub enum BootstrapResult {
//     SingleBootstrap {
//         n_boot: usize,
//         replicas: Vec<f64>,
//         central_val: f64,
//         z: f64,
//         a: f64,
//         median: f64,
//         mean: f64,
//         stddev: f64,
//         ci_68: (f64, f64),
//         ci_95: (f64, f64),
//         ci_99: (f64, f64),
//         failed_samples: usize,
//         histogram: Histogram,
//         central_square: f64,
//         square_error: f64,
//     },
//     DoubleBootstrap(Vec<Vec<f64>>),
// }
#[derive(Debug, Serialize, Deserialize)]
pub enum BootstrapResult {
    Bayesian {
        n_boot: usize,
        replicas: Vec<f64>,
        failed_samples: usize,
        histogram: Histogram,
    },
}
impl BootstrapResult {
    pub fn error(&self) -> f64 {
        match self {
            BootstrapResult::Bayesian {
                replicas: replicas, ..
            } => standard_deviation(&replicas, true),
        }
    }
    pub fn print(&self) -> () {
        println!("{}", serde_json::to_string(&self).unwrap());
    }
}

// impl BootstrapResult {
//     pub fn print(&self) {
//         println!("{}", serde_json::to_string(&self).unwrap());
//     }
//     pub fn get_single_bootstrap_result(self) -> Vec<f64> {
//         match self {
//             BootstrapResult::SingleBootstrap {
//                 n_boot: _,
//                 replicas: v,
//                 central_val: _,
//                 z: _,
//                 a: _,
//                 ci_68: _,
//                 ci_95: _,
//                 ci_99: _,
//                 median: _,
//                 mean: _,
//                 stddev: _,
//                 histogram: _,
//                 failed_samples: _,
//                 central_square: _,
//                 square_error: _,
//             } => v,
//             BootstrapResult::DoubleBootstrap(_) => unimplemented!(),
//         }
//     }
// }
// #[inline(always)]
pub fn get_uniform_weights(length: usize) -> Vec<f64> {
    let mut ans = vec![0.0; length];
    for i in 0..length {
        ans[i] += 1.0 / (length as f64)
    }
    ans
}

// // #[inline(always)]
// pub fn get_subsample(sample: &[f64]) -> Vec<f64> {
//     let mut result = Vec::with_capacity(sample.len());
//     let mut rng = rand::rng();
//     for _ in 0..sample.len() {
//         let index = rng.random_range(..sample.len());
//         result.push(sample[index]);
//     }
//     result
// }

pub fn bayesian_bootstrap<T>(func: T, length: usize, args: &BootstrapArgs) -> BootstrapResult
where
    T: Fn(Vec<f64>) -> Option<f64> + Sync + Send,
{
    let dist = Dirichlet::new(
        &iter::repeat(args.dirichlet_alpha)
            .take(length)
            .collect::<Vec<f64>>(),
    )
    .expect("Could not create Dirichlet distribution.");
    let x = rand::rng().sample(&dist);
    dbg!(&x);
    dbg!(x.iter().sum::<f64>());
    let replicas = drop_nones(
        (0..args.n_boot)
            .into_par_iter()
            .map(|_| func(rand::rng().sample(&dist)))
            .collect(),
    );
    BootstrapResult::Bayesian {
        n_boot: args.n_boot,
        failed_samples: (args.n_boot - replicas.len()),
        histogram: bin(&replicas, args.n_bins_histogram),
        replicas,
    }
}
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

// pub fn bootstrap<T>(func: T, length: usize, boot_args: &BinBootstrapArgs) -> BootstrapResult
// where
//     T: Fn(Vec<f64>) -> Option<f64> + Sync + Send,
// {
//     if let Some(n_boot_double) = boot_args.n_boot_double {
//         BootstrapResult::DoubleBootstrap(
//             (0..n_boot_double)
//                 .into_par_iter()
//                 .map(|_| {
//                     let sample = get_samples(length, boot_args.binwidth);
//                     drop_nones(
//                         (0..boot_args.n_boot)
//                             .map(|_| func(get_subsample(&sample)))
//                             .collect::<Vec<Option<f64>>>(),
//                     )
//                 })
//                 .collect::<Vec<Vec<f64>>>(),
//         )
//     } else {
//         let norm = Normal::standard();
//         let mut replicas = drop_nones(
//             (0..boot_args.n_boot)
//                 .into_par_iter()
//                 .map(|_| func(get_samples(length, boot_args.binwidth)))
//                 .collect(),
//         );
//         replicas.par_sort_unstable_by(f64::total_cmp);
//         let central_val =
//             func((0..length).collect()).expect("Should be able to evaluate the central value!");
//         let mut prop = 0.0;
//         for replica in &replicas {
//             if *replica < central_val {
//                 prop += 1.0
//             } else if *replica == central_val {
//                 prop += 0.5
//             }
//         }
//         let z = norm.inverse_cdf(prop / replicas.len() as f64);

//         // let jack = jackknife_samples(func, length);
//         // let jack_avg = jack.iter().sum::<f64>() / (length as f64);
//         // let mut num = 0.0;
//         // let mut denom = 0.0;

//         // for i in 0..length {
//         // num += (jack_avg - jack[i]).powi(3);
//         // denom += (jack_avg - jack[i]).powi(2);
//         // }

//         // let a = num / (6.0 * denom.powf(3.0 / 2.0));
//         let a = 0.0;
//         let mean = mean(&replicas);
//         let stddev = standard_deviation(&replicas, true);

//         BootstrapResult::SingleBootstrap {
//             ci_68: confidence_interval(&replicas, z, a, 1.0 - 0.682689492137086),
//             ci_95: confidence_interval(&replicas, z, a, 1.0 - 0.954499736103642),
//             ci_99: confidence_interval(&replicas, z, a, 1.0 - 0.997300203936740),
//             n_boot: boot_args.n_boot,
//             median: replicas[replicas.len() / 2],
//             mean: mean,
//             stddev: stddev,
//             histogram: bin(&replicas, boot_args.n_bins_histogram),
//             failed_samples: boot_args.n_boot - replicas.len(),
//             replicas: replicas,
//             central_val: central_val,
//             z: z,
//             a: a,
//             square_error: 2.0 * mean * stddev,
//             central_square: central_val.powi(2),
//         }
//     }
// }
// pub fn jackknife_samples<T>(func: T, length: usize) -> Vec<f64>
// where
//     T: Fn(Vec<usize>) -> Option<f64> + Sync + Send,
// {
//     let mut samples = vec![];
//     let indices: Vec<usize> = (0..length).collect();
//     for i in 0..length {
//         let mut cur_indices = indices.clone();
//         cur_indices.remove(i);
//         samples.push(func(cur_indices).unwrap());
//     }
//     samples
// }

// pub fn confidence_interval(replicas: &[f64], z: f64, a: f64, alpha: f64) -> (f64, f64) {
//     let norm = Normal::standard();
//     let z_a1 = norm.inverse_cdf(alpha / 2.0);
//     let z_a2 = norm.inverse_cdf(1.0 - alpha / 2.0);
//     let lower = norm.cdf(z + (z + z_a1) / (1.0 - a * (z + z_a1)));
//     let upper = norm.cdf(z + (z + z_a2) / (1.0 - a * (z + z_a2)));
//     return (
//         replicas[((replicas.len() as f64) * lower).round() as usize],
//         replicas[((replicas.len() as f64) * upper).round() as usize],
//     );
// }

#[cfg(test)]
#[test]
fn test_bootstrap_samples() {
    assert_eq!(get_uniform_weights(100).len(), 100);
    assert_eq!(get_uniform_weights(10)[1], 0.1);
}
// #[test]
// fn test_jackknife() {
//     assert_eq!(
//         jackknife_samples(
//             |x| {
//                 let vals = vec![1.0, 2.0, 3.0];
//                 let mut acc = 0.0;
//                 for i in &x {
//                     acc += vals[*i]
//                 }
//                 Some(acc / (x.len() as f64))
//             },
//             3
//         ),
//         vec![2.5, 2.0, 1.5]
//     );
// }
#[test]
fn test_bayesian_bootstrap() {
    bayesian_bootstrap(
        |x| {
            let vals = vec![1.0, 2.0, 3.0];
            let mut acc = 0.0;
            for (i, val) in vals.iter().enumerate() {
                acc += x[i] * val;
            }
            Some(acc)
        },
        3,
        &BootstrapArgs {
            n_boot: 1000,
            n_boot_double: None,
            dirichlet_alpha: 1.0,
            n_bins_histogram: 10,
        },
    );
}
