use crate::observables::{Measurement, Observable};
use crate::statistics::{centred_difference_derivative, line_of_best_fit};

#[allow(dead_code)]
pub enum WilsonFlowObservables {
    T,
    E,
    T2E,
    Esym,
    T2Esym,
    TC,
}
impl WilsonFlowObservables {
    //(t,E,t2*Self,Esym,t2*Esym,TC)
    pub fn get_offset(&self) -> usize {
        match self {
            Self::T => 3,
            Self::E => 4,
            Self::T2E => 5,
            Self::Esym => 6,
            Self::T2Esym => 7,
            Self::TC => 8,
        }
    }
}

#[derive(Debug)]
pub struct W {
    pub w: Vec<f64>,
    pub t: Vec<f64>,
}
impl W {
    pub fn new(w: Vec<f64>, t: Vec<f64>) -> Self {
        assert!(w.len() == t.len());
        W { w, t }
    }
}

#[derive(Debug)]
pub struct WilsonFlow {
    pub t: Vec<f64>,
    pub t2_esym: Observable,
    pub tc: Observable,
}
impl WilsonFlow {
    pub fn new(t: Vec<f64>, t2_esym: Observable, tc: Observable) -> Self {
        Self { t, t2_esym, tc }
    }

    pub fn get_subsample_mean_stderr_from_samples(
        &self,
        samples: Vec<usize>,
        channel: WilsonFlowObservables,
    ) -> Measurement {
        match channel {
            WilsonFlowObservables::T2Esym => {
                self.t2_esym.get_subsample_mean_stderr_from_samples(samples)
            }
            WilsonFlowObservables::TC => self.tc.get_subsample_mean_stderr_from_samples(samples),
            _ => unimplemented!(),
        }
    }
    pub fn thermalise(self, thermalisation: usize) -> WilsonFlow {
        WilsonFlow {
            t: self.t,
            t2_esym: self.t2_esym.thermalise(thermalisation),
            tc: self.tc.thermalise(thermalisation),
        }
    }
}
pub fn calculate_w(t2_esym: &[f64], t: &[f64]) -> W {
    let mut w = vec![];
    let dt = t[1] - t[0];
    let d_t2_esym = centred_difference_derivative(t2_esym, dt);
    for i in 0..d_t2_esym.len() {
        w.push(d_t2_esym[i] * t[i + 1]);
    }
    W::new(w, t[1..(t.len() - 1)].to_vec())
}

pub fn calculate_w0(w: W, wref: f64) -> f64 {
    let num_points = 2;
    let pos = w.w.iter().position(|x| *x > wref).unwrap();
    if w.w[pos] == wref {
        w.t[pos - 1].sqrt()
    } else {
        let window = pos - num_points..pos + (num_points - 1);
        let (m, c) = line_of_best_fit(&w.t[window.clone()], &w.w[window]);
        ((wref - c) / m).sqrt()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::load_wf_observables_from_file;
    use crate::observables::get_samples;
    #[test]
    fn calculate_w0_test() {
        let wf = load_wf_observables_from_file("tests/wf_out");
        println!(
            "{:?}",
            calculate_w0(
                calculate_w(
                    &wf.get_subsample_mean_stderr_from_samples(
                        get_samples(wf.tc.nconfs, 1),
                        WilsonFlowObservables::T2Esym,
                    )
                    .values,
                    &wf.t,
                ),
                1.0
            )
        );
    }
}
