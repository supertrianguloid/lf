use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Debug)]
pub struct SingleMeasurement {
    value: f64,
    error: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HistogramRow {
    bin_centre: f64,
    frequency: usize,
}
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
pub fn standard_deviation(values: &[f64], corrected: bool) -> f64 {
    let mean = mean(values);
    (values.par_iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (values.len() as f64 - if corrected { 1.0 } else { 0.0 }))
    .sqrt()
}

pub fn standard_error(values: &[f64]) -> f64 {
    let mean = mean(values);
    values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        .sqrt()
        / values.len() as f64
}
#[allow(dead_code)]
/// Performs the naive error propagation assuming `v1` and `v2` are independent.
pub fn propagate_ratio(v1: SingleMeasurement, v2: SingleMeasurement) -> SingleMeasurement {
    SingleMeasurement {
        value: v1.value / v2.value,
        error: ((v1.error / v1.value).powi(2) + (v2.error / v2.value).powi(2)).sqrt(),
    }
}

pub fn line_of_best_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let xbar = mean(x);
    let ybar = mean(y);
    assert!(x.len() == y.len());
    let mut numerator = 0.0;
    for i in 0..x.len() {
        numerator += (x[i] - xbar) * (y[i] - ybar);
    }
    let mut denominator = 0.0;
    for i in x {
        denominator += (i - xbar).powi(2);
    }
    let m = numerator / denominator;
    (m, ybar - m * xbar)
}

/// Takes (input: `[f64]` of length N, `dx: f64`) and returns `output: Vec<f64>` of length N-2 representing the derivative of the input, where dx is the underlying step size. The result is 'shifted' to the left by 1 due to the endpoints having undefined derivative, so the derivative at `input[1]` is `output[0]` etc.
pub fn centred_difference_derivative(input: &[f64], dx: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(input.len() - 2);
    for i in 1..(input.len() - 1) {
        result.push((input[i + 1] - input[i - 1]) / (2.0 * dx));
    }
    result
}

/// Computes the mean of a set of observables weighted by their errors.
/// [Definition](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Mathematical_definition)
pub fn weighted_mean(sample: &[f64], errors: &[f64]) -> (f64, f64) {
    let weights: Vec<f64> = (0..sample.len())
        .map(|n| 1.0 / errors[n].powf(2.0))
        .collect();
    let mut sum_weight_times_sample = 0.0;
    let mut sum_weights = 0.0;
    for i in 0..sample.len() {
        sum_weight_times_sample += weights[i] * sample[i];
        sum_weights += weights[i];
    }
    (
        sum_weight_times_sample / sum_weights,
        (1.0 / sum_weights).sqrt(),
    )
}

pub fn bin(data: &[f64], nbins: usize) -> Vec<HistogramRow> {
    let lower = data.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let upper = data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let stepsize = (upper - lower) / (nbins as f64);
    let mut bin_ranges = vec![];
    for i in 0..(nbins + 1) {
        bin_ranges.push(lower + (i as f64) * stepsize);
    }
    let mut freq: Vec<usize> = vec![0; nbins];
    freq[0] = data.iter().filter(|x| *x == lower).count();
    let windows: Vec<_> = bin_ranges.windows(2).collect();
    for i in 0..nbins {
        freq[i] = data
            .iter()
            .filter(|x| **x > windows[i][0] && **x <= windows[i][1])
            .count();
    }
    bin_ranges.truncate(nbins);
    let bin_centres: Vec<f64> = bin_ranges.iter().map(|x| x + stepsize / 2.0).collect();
    assert!(freq.len() == bin_ranges.len());
    let mut histogram = vec![];
    for i in 0..freq.len() {
        histogram.push(HistogramRow {
            bin_centre: bin_centres[i],
            frequency: freq[i],
        })
    }
    histogram
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_tests() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(mean(&x), 1.0);
        let x = vec![1.0, 4.0, 2.3, 6.3, -1.5];
        assert_eq!(mean(&x), 2.42);
        let x = vec![
            7.0, 12.0, 5.0, 18.0, 5.0, 9.0, 10.0, 9.0, 12.0, 8.0, 12.0, 16.0,
        ];
        assert_eq!(mean(&x), 10.25);
    }
    #[test]
    fn standard_error_tests() {
        let x = vec![
            7.0, 12.0, 5.0, 18.0, 5.0, 9.0, 10.0, 9.0, 12.0, 8.0, 12.0, 16.0,
        ];
        assert_eq!(standard_error(&x), 1.1063265039459795);
    }
    #[test]
    fn standard_deviation_tests() {
        let x = vec![4.0, 9.0, 11.0, 12.0, 17.0, 5.0, 8.0, 12.0, 14.0];
        assert_eq!(standard_deviation(&x, true), 4.176654695380556);
    }
    #[test]
    fn centred_difference_tests() {
        let x = vec![4.0, 9.0, 4.0, 12.0, 17.0, 5.0, 8.0, 12.0, 14.0];
        assert_eq!(centred_difference_derivative(&x, 1.0).len(), x.len() - 2);
        assert_eq!(centred_difference_derivative(&x, 1.0)[0], 0.0);
    }
    #[test]
    fn weighted_mean_test() {
        let sample = vec![1.0, 2.0];
        let err = vec![0.3, 0.2];
        let w_mean = (1.6923076923076923, 0.16641005886756874);
        assert_eq!(weighted_mean(&sample, &err), w_mean);
    }
    #[test]
    fn propagate_ratio_test() {
        let v1 = SingleMeasurement {
            value: 1.45,
            error: 0.3,
        };
        let v2 = SingleMeasurement {
            value: 3.24,
            error: 0.63,
        };
        assert_eq!(
            propagate_ratio(v1, v2),
            SingleMeasurement {
                value: 0.4475308641975308,
                error: 0.2839274997083719
            }
        )
    }
}
