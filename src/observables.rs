use serde::Serialize;

use crate::bootstrap::get_uniform_weights;
use crate::io::{load_channel_from_file_folded, load_global_t_from_file};
use crate::parser::{BinArgs, HMCArgs};
use crate::statistics::weighted_mean;

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
}

impl ObservableCalculation {
    pub fn load(args: &HMCArgs, channel: String, bin: &BinArgs) -> Self {
        ObservableCalculation {
            obs: load_channel_from_file_folded(&args.filename, &channel)
                .thermalise(args.thermalisation)
                .bin_average(bin.binwidth),
            global_t: load_global_t_from_file(&args.filename),
        }
    }
}

#[derive(Debug, PartialEq)]
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

    // pub fn get_subsample_mean_stderr(&self) -> Measurement {
    // self.get_subsample_mean_stderr_from_weights(&get_weights(self.nconfs))
    // }

    /// Return the weighted mean and error from an array of weights. The array must be the same length as the original data.
    pub fn get_subsample_mean_stderr_from_weights(&self, weights: &[f64]) -> Measurement {
        assert_eq!(self.nconfs, weights.len());
        let mut mu = vec![];
        let mut sigma = vec![];
        for t in 0..(self.each_len) {
            let mut temp = vec![];
            for i in 0..self.nconfs {
                temp.push(self.get_slice(i)[t]);
            }
            let meas = weighted_mean(&temp, weights);
            mu.push(meas.value);
            sigma.push(meas.error);
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

    pub fn get_mean_stderr(&self) -> Measurement {
        self.get_subsample_mean_stderr_from_weights(&get_uniform_weights(self.nconfs))
    }

    pub fn bin_average(self, binsize: usize) -> Self {
        assert!(binsize > 0, "bin size must be > 0");

        let offset = self.nconfs % binsize;
        let out_rows = (self.nconfs - offset) / binsize;
        let mut out = vec![0.0f64; out_rows * self.each_len];

        for b in 0..out_rows {
            for c in 0..self.each_len {
                let mut sum = 0.0f64;
                for r in 0..binsize {
                    let src_row = offset + b * binsize + r;
                    sum += self.data[src_row * self.each_len + c];
                }
                out[b * self.each_len + c] = sum / binsize as f64;
            }
        }
        Self {
            data: out,
            nconfs: out_rows,
            each_len: self.each_len,
        }
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
        dbg!(get_uniform_weights(3));
        assert_eq!(
            o.get_mean_stderr(),
            Measurement::new(vec![2.0, 1.0], vec![0.0, 0.0])
        );
    }
    #[test]
    fn observable_bin_length_one_does_nothing() {
        let obs = Observable {
            each_len: 3,
            nconfs: 2,
            data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        };
        let binned = obs.bin_average(1);
        assert_eq!(
            binned,
            Observable {
                each_len: 3,
                nconfs: 2,
                data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
            }
        );
    }
    #[test]
    fn observable_bin_simple() {
        let obs = Observable {
            each_len: 3,
            nconfs: 2,
            data: vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        };
        let binned = obs.bin_average(2);
        assert_eq!(
            binned,
            Observable {
                each_len: 3,
                nconfs: 1,
                data: vec![2.0, 3.0, 4.0]
            }
        );
    }
    #[test]
    fn observable_bin_g5() {
        let obs = load_channel_from_file_folded("tests/out_test", "g5");

        let binned = obs.bin_average(400);
        dbg!(&binned);
        assert_eq!(
            binned.data,
            vec![
                0.00020271357752861842,
                2.0815550113460623e-5,
                4.963908574852205e-6,
                1.5904561236425944e-6,
                6.060660875220452e-7,
                2.518530740656517e-7,
                1.0936992255071622e-7,
                4.860458723733243e-8,
                2.187598116055109e-8,
                9.908516896138259e-9,
                4.502806185869573e-9,
                2.052097539943126e-9,
                9.389148530294929e-10,
                4.3268002441656626e-10,
                2.0482946938228817e-10,
                1.0866893376096108e-10,
                8.221017894175636e-11
            ]
        );
    }
}
