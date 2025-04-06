use rand::distr::{Distribution, Uniform};
pub fn get_samples(length: usize, binsize: usize) -> Vec<usize> {
    let length_new = length / binsize;
    let mut rng = rand::rng();
    let samples: Vec<_> = Uniform::try_from(0..length)
        .unwrap()
        .sample_iter(&mut rng)
        .take(length_new)
        .collect();
    samples
}

#[cfg(test)]
#[test]
fn test_bootstrap_binning_samples() {
    assert_eq!(get_samples(100, 2).len(), 50);
}
#[test]
fn test_bootstrap_no_binning_samples() {
    assert_eq!(get_samples(100, 1).len(), 100);
}
