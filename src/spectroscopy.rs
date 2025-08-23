use crate::observables::{Measurement, Observable};
use crate::statistics::centred_difference_derivative;
use nalgebra::DVector;
use roots::{find_root_brent, SearchError, SimpleConvergency};
use varpro::prelude::*;
use varpro::problem::*;
use varpro::solvers::levmar::LevMarSolver;

#[derive(Debug)]
struct CoshFit {
    coefficient: f64,
    mass: f64,
}

fn fit_cosh(corr: Measurement, global_t: usize, lower: usize, upper: usize) -> CoshFit {
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

    // the data, weight and initial guesses for our fitting problem
    let t = DVector::from_vec((lower..upper).map(|val| val as f64).collect());
    let y = DVector::from_vec(corr.values[lower..upper].to_vec());
    let w = DVector::from_vec(
        corr.errors
            .into_iter()
            .map(|val| 1.0 / val.powi(2))
            .collect::<Vec<f64>>()[lower..upper]
            .to_vec(),
    );

    let model = SeparableModelBuilder::new(["mass"])
        .independent_variable(t)
        .function(["mass"], generate_models(global_t).0)
        .partial_deriv("mass", generate_models(global_t).1)
        .invariant_function(|t| DVector::from_element(t.len(), 1.))
        // the initial guess for the nonlinear parameters is tau1=1, tau2=5
        .initial_parameters(vec![1.])
        .build()
        .unwrap();

    // describe the fitting problem
    let problem = SeparableProblemBuilder::new(model)
        .observations(y)
        .weights(w)
        .build()
        .unwrap();

    // fit the data
    let fit_result = LevMarSolver::default()
        .fit(problem)
        .expect("fit must succeed");
    // the nonlinear parameters
    let mass = fit_result.nonlinear_parameters();
    // the linear coefficients
    let coeff = fit_result.linear_coefficients().unwrap();
    return CoshFit {
        coefficient: coeff[0],
        mass: mass[0],
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

pub fn effective_pcac_all_t(f_ap: &[f64], f_ps: &[f64]) -> Vec<f64> {
    let d_f_ap = centred_difference_derivative(f_ap, 1.0);
    let mut res = vec![];
    for t in 0..d_f_ap.len() {
        res.push(0.5 * d_f_ap[t] / f_ps[t + 1]);
    }
    res
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
    fn pcac_mass_test() {
        let g5_folded = vec![
            6.46661072864547e-5,
            8.2314099026351e-6,
            2.5820200510923274e-6,
            1.2150389416046531e-6,
            7.240038058465569e-7,
            4.834076999787681e-7,
            3.410793027523067e-7,
            2.477269442682027e-7,
            1.8270878598292142e-7,
            1.360036393137522e-7,
            1.0177445902377655e-7,
            7.640036563239633e-8,
            5.743429959258569e-8,
            4.3239346295428004e-8,
            3.2564601272771856e-8,
            2.4579454044319454e-8,
            1.861013111382292e-8,
            1.4140445525124136e-8,
            1.0813029175351652e-8,
            8.357054040587225e-9,
            6.571215520943662e-9,
            5.314192827473403e-9,
            4.481519311445979e-9,
            4.008671232799453e-9,
            3.855716132260987e-9,
        ];
        let g5_g0g5_folded = vec![
            7.648241073344489e-10,
            -2.3218763553626026e-6,
            -4.718504157990981e-7,
            -1.9539584803061266e-7,
            -1.1795798865059264e-7,
            -8.164369255127817e-8,
            -5.921349145094337e-8,
            -4.3759922681291586e-8,
            -3.258560330782088e-8,
            -2.4401210344785395e-8,
            -1.829336499864451e-8,
            -1.3786895148330275e-8,
            -1.0380977738554743e-8,
            -7.827351111015616e-9,
            -5.88095185702586e-9,
            -4.422898137040417e-9,
            -3.314551154308905e-9,
            -2.477083469640234e-9,
            -1.8395964605396757e-9,
            -1.3521991607241546e-9,
            -9.689828352753765e-10,
            -6.682199016286922e-10,
            -4.168739497298201e-10,
            -2.00372819631762e-10,
            3.744202576004903e-12,
        ];
        let m_ps = vec![
            2.0612799075393324,
            1.1593852544934382,
            0.7537959311332677,
            0.5177347575523663,
            0.40393637164531115,
            0.34874715873675965,
            0.31979686034049276,
            0.3044603352624457,
            0.29527531895135056,
            0.2900532014199751,
            0.28702235843395774,
            0.28580244454001297,
            0.2847162359878983,
            0.2849755804907339,
            0.28391648217465176,
            0.28284397130201583,
            0.2827881028788996,
            0.28246711108017003,
            0.28216571744378305,
            0.28212212500308353,
            0.28175102053342865,
            0.2818170004434757,
            0.28130289770125855,
            0.2807495623706952,
        ];
        assert_eq!(
            // effective_pcac_all_t(&g5_g0g5_folded, &g5_folded, &m_ps),
            effective_pcac_all_t(&g5_g0g5_folded, &g5_folded),
            vec![2.0, 3.0, 4.0]
        );
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
        let fit = fit_cosh(g5, 10, 0, 5);
        assert!(fit.coefficient - 1.0 < f64::EPSILON);
        assert!(fit.mass - 1.0 < f64::EPSILON);
    }
}
