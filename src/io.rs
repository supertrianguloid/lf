use crate::statistics::centred_difference_derivative;
use crate::statistics::{mean, standard_error};
use rand::distributions::{Distribution, Uniform};
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

pub enum WfObservable {
    T,
    E,
    T2E,
    ESym,
    T2Esym,
    Tc,
}

impl WfObservable {
    pub fn get_offset(&self) -> usize {
        match self {
            WfObservable::T => 0,
            WfObservable::E => 1,
            WfObservable::T2E => 2,
            WfObservable::ESym => 3,
            WfObservable::T2Esym => 4,
            WfObservable::Tc => 5,
        }
    }
}

pub struct Observable {
    each_len: usize,
    nconfs: usize,
    data: Vec<f64>,
}

impl Observable {
    /// Returns the inner data at the given configuration number as a slice
    pub fn get_slice(&self, conf_no: usize) -> &[f64] {
        &self.data[self.each_len * conf_no..self.each_len * (conf_no + 1)]
    }
    pub fn thermalise(mut self, thermalisation: usize) -> Observable {
        return Observable {
            data: self.data.split_off(thermalisation * self.each_len),
            nconfs: self.nconfs - thermalisation,
            ..self
        };
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
    pub fn get_subsample_mean_stderr(&self, binsize: usize) -> (Vec<f64>, Vec<f64>) {
        let length_new = self.nconfs / binsize;
        let mut mu = vec![];
        let mut sigma = vec![];
        let mut rng = rand::thread_rng();
        let samples: Vec<_> = Uniform::from(0..self.nconfs)
            .sample_iter(&mut rng)
            .take(length_new)
            .collect();
        for t in 0..(self.each_len) {
            let mut temp = vec![];
            for sample in samples.iter() {
                temp.push(self.get_slice(*sample)[t]);
            }
            let mean = mean(&temp);
            mu.push(mean);
            sigma.push(standard_error(&temp));
        }
        (mu, sigma)
    }
}

/// Folds a Vec<f64> of length N into a correlator of length N/2 + 1, ignoring the first point.
pub fn fold_correlator(mut corr: Vec<f64>, symmetric: bool) -> Vec<f64> {
    for i in 1..((corr.len() / 2) + 1) {
        corr[i] = 0.5 * (corr[i] + corr[corr.len() - i] * if symmetric { 1.0 } else { -1.0 });
    }
    corr.truncate(corr.len() / 2 + 1);
    corr.shrink_to_fit();
    corr
}

/// TODO: Speed this up by moving the flatten earlier.
pub fn load_channel_from_file_folded(hmc_filename: &str, channel: &str) -> Observable {
    fn load_channel(hmc_filename: &str, channel: &str, fold_sign: bool) -> Observable {
        let data = BufReader::new(File::open(hmc_filename).unwrap())
            .lines()
            .map(|line| line.unwrap())
            .filter(|line| line.contains("DEFAULT_SEMWALL TRIPLET"))
            .filter(|line| line.contains(channel))
            .map(|line| {
                line.trim()
                    .split(" ")
                    .skip(6)
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>()
            })
            .map(|corr| fold_correlator(corr, fold_sign))
            .collect::<Vec<Vec<f64>>>();
        Observable {
            each_len: data[1].len(),
            nconfs: data.len(),
            data: data.into_iter().flatten().collect(),
        }
    }
    match channel {
        "g5" => load_channel(hmc_filename, channel, true),
        "id" => load_channel(hmc_filename, channel, true),
        "gk" => load_channel(hmc_filename, "g1", true).average_with(
            load_channel(hmc_filename, "g2", true),
            load_channel(hmc_filename, "g3", true),
        ),
        "g5gk" => load_channel(hmc_filename, "g5g1", true).average_with(
            load_channel(hmc_filename, "g5g2", true),
            load_channel(hmc_filename, "g5g3", true),
        ),
        _ => todo!("Unknown channel"),
    }
}

pub fn load_global_t_from_file(hmc_filename: &str) -> usize {
    BufReader::new(File::open(hmc_filename).unwrap())
        .lines()
        .map(|line| line.unwrap())
        .find(|line| line.contains("Global size is"))
        .unwrap()
        .split(" ")
        .nth(3)
        .unwrap()
        .trim()
        .split("x")
        .next()
        .unwrap()
        .parse::<usize>()
        .unwrap()
}
pub fn load_wf_observable_from_file(
    wf_filename: &str,
    channel: WfObservable,
    nmeas: usize,
) -> Observable {
    let mut obs: Vec<f64> = vec![];
    //(t,E,t2*E,Esym,t2*Esym,TC)
    for line in BufReader::new(File::open(wf_filename).unwrap())
        .lines()
        .map(|line| line.unwrap())
        .filter(|line| line.contains("WILSONFLOW"))
    {
        let line: Vec<_> = line.split(" ").collect();
        obs.push(line[3 + channel.get_offset()].parse::<f64>().unwrap());
    }
    Observable {
        each_len: nmeas,
        nconfs: obs.len() / nmeas,
        data: obs,
    }
}

pub fn calculate_w(T2Esym: &Vec<f64>, t: &Vec<f64>, dt: f64) -> Vec<f64> {
    let mut ans = vec![];
    let dT2ESym = centred_difference_derivative(T2Esym, dt);
    for i in 0..dT2ESym.len() {
        ans.push(dT2ESym[i] * t[i + 1]);
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_global_t_test() {
        assert_eq!(load_global_t_from_file("tests/out_test"), 32);
    }
    #[test]
    fn observable_slice_test() {
        let mut obs = Observable {
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
    fn folding_tests() {
        let x = vec![1.0, 2.0, 3.0, 14.0, 5.0, 6.0];
        assert_eq!(fold_correlator(x, true), vec![1.0, 4.0, 4.0, 14.0]);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(fold_correlator(x, true), vec![1.0, 3.0, 3.0]);
        let x = vec![1.0, 2.0, 3.0, 4.5, 5.0];
        assert_eq!(fold_correlator(x, true), vec![1.0, 3.5, 3.75]);
        let x = vec![1.0, 2.0, 3.0, 2.0, 5.0];
        assert_eq!(fold_correlator(x, false), vec![1.0, -1.5, 0.5]);
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
            (vec![2.0, 1.0], vec![0.0, 0.0])
        );
    }
    #[test]
    fn load_wilsonflow_test() {
        let wf_t = load_wf_observable_from_file("tests/wf_out", WfObservable::T, 1000);
        assert_eq!(wf_t.nconfs, 275);
        assert_eq!(wf_t.each_len, 1000);
        assert_eq!(wf_t.data[2], 2e-1);
        let wf_esym = load_wf_observable_from_file("tests/wf_out", WfObservable::ESym, 1000);
        assert_eq!(wf_esym.nconfs, 275);
        assert_eq!(wf_esym.each_len, 1000);
        assert_eq!(wf_esym.data[3], 3.0803808637068719e-01);
    }
    #[test]
    fn calculate_w0_test() {
        let wf_t2esym = load_wf_observable_from_file("tests/wf_out", WfObservable::T2Esym, 1000);
        let wf_t = load_wf_observable_from_file("tests/wf_out", WfObservable::T, 1000);
        todo!();
        //assert!(find_w0(wf_esym) - 5.211 < 0.0001);
    }
}
