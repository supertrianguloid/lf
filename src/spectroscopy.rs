// use nalgebra::DVector;
use roots::{find_root_brent, SearchError, SimpleConvergency};
// use varpro::model::SeparableModel;
// use varpro::prelude::*;
// use varpro::solvers::levmar::{FitResult, LevMarProblemBuilder, LevMarSolver};

//fn fit_exponential(corr: Correlator) -> FitResult<SeparableModel<f64>, false> {
//fn exp_decay(t: &DVector<f64>, m: f64) -> DVector<f64> {
//t.map(|t| (-m * t).exp())
//}
//fn exp_decay_dm(t: &DVector<f64>, m: f64) -> DVector<f64> {
//t.map(|t| -t * (-m * t).exp())
//}
//let model = SeparableModelBuilder::<f64>::new(&["m1"])
//.function(&["m1"], exp_decay)
//.partial_deriv("m1", exp_decay_dm)
//.invariant_function(|x| DVector::from_element(x.len(), 1.))
//.independent_variable(corr.t)
//.initial_parameters(vec![1.])
//.build()
//.unwrap();

//let problem = LevMarProblemBuilder::new(model)
//.observations(corr.values)
//.weights(corr.errors.map(|e| 1.0 / e.powf(2.0)))
//.build()
//.unwrap();

//LevMarSolver::default()
//.fit(problem)
//.expect("fit must succeed")
//}

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
}
