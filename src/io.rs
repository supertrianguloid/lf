use crate::observables::Observable;
use crate::wilsonflow::{WilsonFlow, WilsonFlowObservables};
use std::fs::{read_to_string, File};

use std::io::{BufRead, BufReader};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum SymmetryType {
    Symmetric,
    Antisymmetric,
}
/// Folds a `Vec<f64>` of length N into a correlator of length N/2 + 1, ignoring the first point.
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
        let channel = format!(" {}=", channel);
        let data = BufReader::new(File::open(hmc_filename).unwrap())
            .lines()
            .map(|line| line.unwrap())
            .filter(|line| line.contains("DEFAULT_SEMWALL TRIPLET"))
            .filter(|line| line.contains(&channel))
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
pub fn load_wf_observables_from_file(wf_filename: &str) -> WilsonFlow {
    let mut t: Vec<_> = vec![];
    let mut t2_esym: Vec<_> = vec![];
    let mut tc: Vec<_> = vec![];
    let wf_file = read_to_string(wf_filename).unwrap();
    let mut minlength = usize::MAX;
    for block in wf_file
        .split("[IO][0]SU2 quaternion representation")
        .skip(1)
    {
        // println!("{}", block);
        // println!("NEXT BLOCK");
        let mut t_l: Vec<f64> = vec![];
        let mut t2_esym_l: Vec<f64> = vec![];
        let mut tc_l: Vec<f64> = vec![];

        for line in block.lines().filter(|line| line.contains("WILSONFLOW")) {
            let line: Vec<_> = line.split(" ").collect();
            t_l.push(
                line[WilsonFlowObservables::T.get_offset()]
                    .parse::<f64>()
                    .unwrap(),
            );
            t2_esym_l.push(
                line[WilsonFlowObservables::T2Esym.get_offset()]
                    .parse::<f64>()
                    .unwrap(),
            );
            tc_l.push(
                line[WilsonFlowObservables::TC.get_offset()]
                    .parse::<f64>()
                    .unwrap(),
            );
        }
        if t_l.len() < minlength {
            minlength = t_l.len();
        }
        t.push(t_l);
        t2_esym.push(t2_esym_l);
        tc.push(tc_l);
    }

    let mut tc_g = vec![];
    let mut t2_esym_g = vec![];

    for i in 0..t.len() {
        t[i].truncate(minlength);
        t2_esym[i].truncate(minlength);
        tc[i].truncate(minlength);
    }
    for i in 0..t.len() {
        t2_esym[i].iter().for_each(|elem| t2_esym_g.push(*elem));
        tc[i].iter().for_each(|elem| tc_g.push(*elem));
    }
    assert!(t.iter().all(|w| { w == &t[0] }));

    WilsonFlow::new(
        t[0].clone(),
        Observable::new(minlength, t.len(), t2_esym_g),
        Observable::new(minlength, t.len(), tc_g),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_global_t_test() {
        assert_eq!(load_global_t_from_file("tests/out_test"), 32);
    }
    #[test]
    fn load_file_test() {
        let channel = load_channel_from_file_folded("tests/out_test", "g5");
        assert_eq!(channel.each_len, 17);
        assert_eq!(channel.nconfs, 499);
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
    #[test]
    fn load_wilsonflow_test() {
        let wf = load_wf_observables_from_file("tests/wf_out");
        assert_eq!(wf.t.len(), 1001);
        assert_eq!(wf.t[2], 2e-1);
        assert_eq!(wf.t2_esym.nconfs, 275);
        assert_eq!(wf.t2_esym.data[3], 2.7723427773361856e-02);
    }
}
