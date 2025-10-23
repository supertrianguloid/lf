use crate::observables::{Measurement, Observable};
use crate::statistics::centred_difference_derivative;
use nalgebra::DVector;
use roots::{find_root_brent, SearchError, SimpleConvergency};
use varpro::prelude::*;
use varpro::problem::*;
use varpro::solvers::levmar::LevMarSolver;

#[derive(Debug)]
pub struct CoshFit {
    pub coefficient: f64,
    pub mass: f64,
}

pub fn fit_cosh(corr: &Measurement, global_t: usize, lower: usize, upper: usize) -> CoshFit {
    // build the model
    fn generate_models(
        global_t: usize,
    ) -> (
        impl Fn(&DVector<f64>, f64) -> DVector<f64> + Sync + Send,
        impl Fn(&DVector<f64>, f64) -> DVector<f64> + Sync + Send,
    ) {
        return (
            move |t: &DVector<f64>, m: f64| t.map(|t| f64::cosh(m * (t - (global_t as f64) / 2.0))),
            move |t: &DVector<f64>, m: f64| {
                t.map(|t| {
                    (t - (global_t as f64) / 2.0) * f64::sinh(m * (t - (global_t as f64) / 2.0))
                })
            },
        );
    }

    let t = DVector::from_vec((lower..upper).map(|val| val as f64).collect());
    let y = DVector::from_vec(corr.values[lower..upper].to_vec());
    let w = DVector::from_vec(
        corr.errors
            .iter()
            .map(|val| 1.0 / val.powi(2))
            .collect::<Vec<f64>>()[lower..upper]
            .to_vec(),
    );

    let model = SeparableModelBuilder::new(["mass"])
        .independent_variable(t)
        .function(["mass"], generate_models(global_t).0)
        .partial_deriv("mass", generate_models(global_t).1)
        .invariant_function(|t| DVector::from_element(t.len(), 1.))
        .initial_parameters(vec![1.])
        .build()
        .unwrap();

    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .weights(w)
        .build()
        .unwrap();

    let fit_result = LevMarSolver::default()
        .fit(problem)
        .expect("fit must succeed");
    let mass = fit_result.nonlinear_parameters();
    let coeff = fit_result.linear_coefficients().unwrap();
    return CoshFit {
        coefficient: coeff[0],
        mass: mass[0].abs(),
    };
}

fn find_range<F>(lower: f64, higher: f64, f: F) -> Option<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    fn triangle(x: usize) -> usize {
        (x.pow(2) + x) / 2
    }
    let maxiters = 10;
    let mut diff = higher - lower;
    for i in 1..maxiters {
        for j in 0..=triangle(i) {
            let j = j as f64;
            if f(lower + diff * j) * f(lower + diff * (j + 1.0)) < 0.0 {
                return Some(((lower + diff * j), (lower + diff * (j + 1.0))));
            }
        }
        diff /= 2.0;
    }
    None
}

pub fn effective_pcac(f_ap: &[f64], f_ps: &[f64], m_ps: &[f64], t: usize) -> f64 {
    let d_f_ap = centred_difference_derivative(f_ap, 1.0);
    0.5 * (m_ps[t + 1] / m_ps[t + 1].sinh()) * d_f_ap[t] / f_ps[t + 1]
}

pub fn effective_mass_all_t(
    correlator: &[f64],
    global_t: usize,
    t_min: usize,
    t_max: usize,
    solver_precision: f64,
) -> Option<Vec<f64>> {
    let mut result = vec![];
    for tau in t_min..(t_max + 1) {
        let mass = effective_mass(correlator, global_t, tau, solver_precision);
        match mass {
            Err(_) => return None,
            Ok(val) => result.push(val),
        };
    }
    Some(result)
}

///Computes the effective mass in a correlator. Tau must be >= 1.
pub fn effective_mass(
    correlator: &[f64],
    global_t: usize,
    tau: usize,
    solver_precision: f64,
) -> Result<f64, SearchError> {
    let mut convergency = SimpleConvergency {
        eps: solver_precision,
        max_iter: 300000,
    };
    let f = |m| eff_mass_eq(correlator, tau, global_t, m);
    let range = find_range(0.0, 100.0, f).ok_or(SearchError::NoBracketing)?;
    find_root_brent(range.0, range.1, f, &mut convergency)
}

fn eff_mass_h(global_t: usize, tau: usize, e1: f64, e2: f64) -> f64 {
    let tau = tau as f64;
    let global_t = global_t as f64;
    (-e1 * tau - e2 * (global_t - tau)).exp() + (-e2 * tau - e1 * (global_t - tau)).exp()
}

fn eff_mass_eq(correlator: &[f64], tau: usize, global_t: usize, m: f64) -> f64 {
    eff_mass_h(global_t, tau - 1, 0.0, m) / eff_mass_h(global_t, tau, 0.0, m)
        - correlator[tau - 1] / correlator[tau]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eff_mass_h_tests() {
        assert_eq!(eff_mass_h(10, 2, 3.0, 6.0), 2.3195228655698555e-16);
        assert_eq!(eff_mass_h(4, 3, 2.456, 2.11), 0.00022937499702432347);
    }
    #[test]
    fn eff_mass_eq_tests() {
        let corr1 = vec![
            6.999835709325914e-5,
            1.1392827663242467e-5,
            4.5063197639444055e-6,
            2.4627705990121428e-6,
            1.5216356567508807e-6,
            1.0348172391553668e-6,
            7.372955580927013e-7,
            5.345199694705964e-7,
            3.9276383154187864e-7,
            2.9393756701261805e-7,
            2.229382882244821e-7,
            1.7140717951704845e-7,
            1.3276004858587368e-7,
            1.0150142065073279e-7,
            7.613972238897562e-8,
            5.747611761369169e-8,
            4.3529783269641134e-8,
            3.339934104614167e-8,
            2.5739422230174207e-8,
            1.9687721674438752e-8,
            1.5448966740457453e-8,
            1.255453638263246e-8,
            1.0594169570805888e-8,
            9.467287963230538e-9,
            9.088527766907613e-9,
        ];
        let t = 48;
        assert_eq!(eff_mass_eq(&corr1, 3, t, 2.0), 5.559279600156479);
        assert_eq!(eff_mass_eq(&corr1, 9, t, 4.0), 53.2619348785668);
    }
    #[test]
    fn cosh_fit_test() {
        let g5 = Measurement {
            values: vec![
                74.20994852478785,
                27.308232836016487,
                10.067661995777765,
                3.7621956910836314,
                1.5430806348152437,
                1.0,
            ],
            errors: vec![
                0.030234548959598145,
                0.02690728250565751,
                0.028623548152925915,
                0.028724017971541593,
                0.030365329860488266,
                0.02813898993584893,
            ],
        };
        let fit = fit_cosh(&g5, 10, 0, 5);
        assert!(fit.coefficient - 1.0 < f64::EPSILON);
        assert!(fit.mass - 1.0 < f64::EPSILON);
    }
}
