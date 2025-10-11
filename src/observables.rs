use crate::bootstrap::get_samples;
use crate::io::{load_channel_from_file_folded, load_global_t_from_file};
use crate::parser::HMCArgs;
use crate::statistics::{mean, standard_error};

#[derive(PartialEq, Debug)]
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
    pub fn load(args: &HMCArgs, channel: String) -> Self {
        ObservableCalculation {
            obs: load_channel_from_file_folded(&args.filename, &channel)
                .thermalise(args.thermalisation),
            global_t: load_global_t_from_file(&args.filename),
        }
    }
}

#[derive(Debug)]
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
