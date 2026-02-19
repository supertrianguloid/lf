use serde::Serialize;

// use crate::bootstrap::get_samples;
use crate::io::{load_channel_from_file_folded, load_global_l_from_file, load_global_t_from_file};
use crate::parser::{BootstrapArgs, HMCArgs};
use crate::statistics::{mean, standard_error};
use booted::bootstrap::{Bootstrap, BootstrapResult, Estimator};
use booted::samplers::SamplingStrategy;
use booted::summary::{BootstrapSummary, Summarizable, SummaryStatistic};
use serde_json::to_string;

use rand::distr::{Distribution, Uniform};

#[derive(PartialEq, Debug, Serialize)]
pub struct Measurement {
    pub values: Vec<f64>,
    pub errors: Vec<f64>,
}
impl Measurement {
    pub fn new(values: Vec<f64>, errors: Vec<f64>) -> Self {
        Self { values, errors }
    }
}

#[derive(Debug)]
pub struct ObservableCalculation {
    pub obs: Observable,
    pub global_t: usize,
    pub global_l: usize,
}
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

pub fn bootstrap<F, T>(estimator: Estimator<F>, args: BootstrapArgs)
where
    F: Fn(&[usize]) -> Option<T> + Send + Sync + Clone + 'static,
    T: SummaryStatistic,
{
    let sampler = match args.strategy {
        crate::parser::BlockStrategy::Blocking => SamplingStrategy::Block {
            block_size: args.blocksize,
        },
        crate::parser::BlockStrategy::Thinning => SamplingStrategy::MOutOfN {
            m: estimator.indices().len() / args.blocksize,
        },
    };
    if let Some(n_boot_double) = args.n_boot_double {
        let est = estimator.clone();
        let sampler_inner = sampler.clone();
        let outer_estimator = Estimator::new()
            .indices(estimator.indices().to_owned())
            .from(move |indices: &[usize]| {
                let inner_estimator = est.clone().set_indices(indices.to_owned());

                let inner_result: BootstrapResult<T> = Bootstrap::builder()
                    .n_boot(args.n_boot)
                    .sampler(sampler_inner.clone())
                    .estimator(inner_estimator)
                    .build()
                    .run();

                let summary: BootstrapSummary<T> = inner_result.summarize();

                // Now T::standard_error works because T: SummaryStatistic
                Some(T::standard_error(&summary.statistics))
            })
            .build();

        println!(
            "{}",
            to_string(
                &Bootstrap::builder()
                    .n_boot(n_boot_double)
                    .estimator(outer_estimator)
                    .sampler(sampler.clone())
                    .build()
                    .run()
                    .summarize()
            )
            .unwrap()
        );
    } else {
        // --- Single Bootstrap ---
        println!(
            "{}",
            to_string(
                &Bootstrap::builder()
                    .n_boot(args.n_boot)
                    .sampler(sampler.clone())
                    .estimator(estimator)
                    .build()
                    .run()
                    .summarize()
            )
            .unwrap()
        );
    }
}

impl ObservableCalculation {
    pub fn load(args: &HMCArgs, channel: String) -> Self {
        ObservableCalculation {
            obs: load_channel_from_file_folded(&args.filename, &channel)
                .thermalise(args.thermalisation),
            global_t: load_global_t_from_file(&args.filename),
            global_l: load_global_l_from_file(&args.filename),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Observable {
    pub each_len: usize,
    pub nconfs: usize,
    pub data: Vec<f64>,
}
impl Observable {
    /// Returns the inner data at the given configuration number as a slice
    pub fn get_slice(&self, conf_no: usize) -> &[f64] {
        &self.data[self.each_len * conf_no..self.each_len * (conf_no + 1)]
    }
    pub fn thermalise(mut self, thermalisation: usize) -> Observable {
        Observable {
            data: self.data.split_off(thermalisation * self.each_len),
            nconfs: self.nconfs - thermalisation,
            ..self
        }
    }
    pub fn average_with(self, o2: Observable, o3: Observable) -> Observable {
        let mut new_data = Vec::with_capacity(self.data.len());
        assert!(self.each_len == o2.each_len && o2.each_len == o3.each_len);
        assert!(self.nconfs == o2.nconfs && o2.nconfs == o3.nconfs);
        for i in 0..self.data.len() {
            new_data.push((self.data[i] + o2.data[i] + o3.data[i]) / 3.0);
        }
        Observable {
            data: new_data,
            ..self
        }
    }

    pub fn get_subsample_mean_stderr(&self, binsize: usize) -> Measurement {
        self.get_subsample_mean_stderr_from_samples(&get_samples(self.nconfs, binsize))
    }

    pub fn get_subsample_mean_stderr_from_samples(&self, samples: &[usize]) -> Measurement {
        let mut mu = vec![];
        let mut sigma = vec![];
        for t in 0..(self.each_len) {
            let mut temp = vec![];
            for sample in samples.iter() {
                temp.push(self.get_slice(*sample)[t]);
            }
            let mean = mean(&temp);
            mu.push(mean);
            sigma.push(standard_error(&temp));
        }
        Measurement::new(mu, sigma)
    }

    pub fn new(each_len: usize, nconfs: usize, data: Vec<f64>) -> Observable {
        Observable {
            each_len,
            nconfs,
            data,
        }
    }

    // pub fn block_average(mut self, blocksize: usize) -> Observable {
    //     let offset = self.nconfs % blocksize;

    //     let obs = Observable {
    //         data: self.data.split_off(offset * self.each_len),
    //         nconfs: self.nconfs - offset,
    //         each_len: self.each_len,
    //     };

    //     let mut new_data = Vec::with_capacity(obs.nconfs / blocksize * obs.each_len);
    //     let num_blocks = obs.nconfs / blocksize;

    //     for block in 0..num_blocks {
    //         let mut block_sum = vec![0.0; obs.each_len];
    //         for i in 0..blocksize {
    //             let config_data = obs.get_slice(block * blocksize + i);
    //             for (j, v) in config_data.iter().enumerate() {
    //                 block_sum[j] += v;
    //             }
    //         }
    //         for sum in block_sum.into_iter() {
    //             new_data.push(sum / blocksize as f64);
    //         }
    //     }
    //     Observable {
    //         nconfs: num_blocks,
    //         each_len: self.each_len,
    //         data: new_data,
    //     }
    // }

    pub fn get_mean_stderr(&self) -> Measurement {
        self.get_subsample_mean_stderr_from_samples(&(0..(self.nconfs - 1)).collect::<Vec<usize>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observable_slice_test() {
        let obs = Observable {
            each_len: 2,
            nconfs: 3,
            data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        };
        assert_eq!(obs.get_slice(0), &vec![1.0, 2.0]);
        assert_eq!(obs.get_slice(1), &vec![3.0, 3.0]);
        assert_eq!(obs.get_slice(2), &vec![4.0, 5.0]);
        let obs = obs.thermalise(2);
        assert_eq!(obs.get_slice(0), &vec![4.0, 5.0]);
    }
    // #[test]
    // fn observable_blocksize_one_does_nothing_test() {
    //     let obs = Observable {
    //         each_len: 2,
    //         nconfs: 3,
    //         data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    //     };
    //     assert_eq!(&obs.clone().block_average(1).data, &obs.data);
    // }
    // #[test]
    // fn observable_blocksize_two_test() {
    //     let obs = Observable {
    //         each_len: 2,
    //         nconfs: 3,
    //         data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    //     };
    //     assert_eq!(&obs.clone().block_average(2).data, &vec![3.5, 4.0]);
    // }
    #[test]
    fn average_observable_test() {
        let obs = Observable {
            each_len: 2,
            nconfs: 3,
            data: vec![1.0, 1.0, 3.0, 3.0, 4.0, 5.0],
        };
        let o2 = Observable {
            data: vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
            ..obs
        };
        let o3 = Observable {
            data: vec![4.0, 10.0, 2.0, 2.0, 3.0, 4.0],
            ..obs
        };
        let avg = obs.average_with(o2, o3);
        assert_eq!(avg.get_slice(0), &vec![2.0, 4.0]);
    }
    #[test]
    fn subsample_tests() {
        let o = Observable {
            each_len: 2,
            nconfs: 3,
            data: vec![2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        };
        assert_eq!(
            o.get_subsample_mean_stderr(1),
            Measurement::new(vec![2.0, 1.0], vec![0.0, 0.0])
        );
    }
}
