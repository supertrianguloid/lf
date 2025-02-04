use crate::observables::Observable;
use crate::statistics::centred_difference_derivative;
use crate::wilsonflow::WfObservables;
use std::fs::{read_to_string, File};
use std::io::{BufRead, BufReader};

#[derive(Clone, Copy)]
pub enum SymmetryType {
    Symmetric,
    Antisymmetric,
}

pub struct ObservablePair {
    each_len_1: usize,
    each_len_2: usize,
    nconfs: usize,
    data_1: Vec<f64>,
    data_2: Vec<f64>,
}
/// Folds a Vec<f64> of length N into a correlator of length N/2 + 1, ignoring the first point.
pub fn fold_correlator(mut corr: Vec<f64>, symmetry: SymmetryType) -> Vec<f64> {
    for i in 1..((corr.len() / 2) + 1) {
        corr[i] = 0.5
            * (corr[i]
                + corr[corr.len() - i]
                    * match symmetry {
                        SymmetryType::Symmetric => 1.0,
                        SymmetryType::Antisymmetric => -1.0,
                    });
    }
    corr.truncate(corr.len() / 2 + 1);
    corr.shrink_to_fit();
    corr
}

pub fn load_channel_from_file_folded(hmc_filename: &str, channel: &str) -> Observable {
    fn load_channel(hmc_filename: &str, channel: &str, symmetry: SymmetryType) -> Observable {
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
            .map(|corr| fold_correlator(corr, symmetry))
            .collect::<Vec<Vec<f64>>>();
        Observable::new(
            data[1].len(),
            data.len(),
            data.into_iter().flatten().collect(),
        )
    }
    match channel {
        "g5" => load_channel(hmc_filename, channel, SymmetryType::Symmetric),
        "id" => load_channel(hmc_filename, channel, SymmetryType::Symmetric),
        "gk" => load_channel(hmc_filename, "g1", SymmetryType::Symmetric).average_with(
            load_channel(hmc_filename, "g2", SymmetryType::Symmetric),
            load_channel(hmc_filename, "g3", SymmetryType::Symmetric),
        ),
        "g5gk" => load_channel(hmc_filename, "g5g1", SymmetryType::Symmetric).average_with(
            load_channel(hmc_filename, "g5g2", SymmetryType::Symmetric),
            load_channel(hmc_filename, "g5g3", SymmetryType::Symmetric),
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
// pub fn load_wf_observables_from_file(wf_filename: &str) -> Observable {
//     let mut obs: Vec<f64> = vec![];
//     let wf_file = read_to_string(wf_filename).unwrap();
//     for line in BufReader::new(File::open(wf_filename).unwrap())
//         .lines()
//         .map(|line| line.unwrap())
//         .filter(|line| line.contains("WILSONFLOW"))
//     {
//         let line: Vec<_> = line.split(" ").collect();
//         obs.push(line[3 + channel.get_offset()].parse::<f64>().unwrap());
//     }
// }

pub fn calculate_w(t2_Esym: &Vec<f64>, t: &Vec<f64>) -> Vec<f64> {
    let mut w = vec![];
    let dt = t[1] - t[0];
    let d_t2_Esym = centred_difference_derivative(t2_Esym, dt);
    for i in 0..d_t2_Esym.len() {
        w.push(d_t2_Esym[i] * t[i + 1]);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_global_t_test() {
        assert_eq!(load_global_t_from_file("tests/out_test"), 32);
    }
    #[test]
    fn folding_tests() {
        let x = vec![1.0, 2.0, 3.0, 14.0, 5.0, 6.0];
        assert_eq!(
            fold_correlator(x, SymmetryType::Symmetric),
            vec![1.0, 4.0, 4.0, 14.0]
        );
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            fold_correlator(x, SymmetryType::Symmetric),
            vec![1.0, 3.0, 3.0]
        );
        let x = vec![1.0, 2.0, 3.0, 4.5, 5.0];
        assert_eq!(
            fold_correlator(x, SymmetryType::Symmetric),
            vec![1.0, 3.5, 3.75]
        );
        let x = vec![1.0, 2.0, 3.0, 2.0, 5.0];
        assert_eq!(
            fold_correlator(x, SymmetryType::Antisymmetric),
            vec![1.0, -1.5, 0.5]
        );
    }
    // #[test]
    // fn load_wilsonflow_test() {
    //     let wf_t = load_wf_observable_from_file("tests/wf_out", WfObservables::t, 1000);
    //     assert_eq!(wf_t.nconfs, 275);
    //     assert_eq!(wf_t.each_len, 1000);
    //     assert_eq!(wf_t.data[2], 2e-1);
    //     let wf_esym = load_wf_observable_from_file("tests/wf_out", WfObservables::Esym, 1000);
    //     assert_eq!(wf_esym.nconfs, 275);
    //     assert_eq!(wf_esym.each_len, 1000);
    //     assert_eq!(wf_esym.data[3], 3.0803808637068719e-01);
    // }
    // #[test]
    // fn calculate_w0_test() {
    //     let wf_t2esym = load_wf_observable_from_file("tests/wf_out", WfObservable::t2_Esym, 1000);
    //     let wf_t = load_wf_observable_from_file("tests/wf_out", WfObservable::t, 1000);
    // dbg!(calculate_w(
    //     &wf_t2esym.get_subsample_mean_stderr(100).0,
    //     &wf_t.get_subsample_mean_stderr(100).0
    // ));
    // dbg!(wf_t.data);
    //assert!(find_w0(wf_esym) - 5.211 < 0.0001);
    // }
}
