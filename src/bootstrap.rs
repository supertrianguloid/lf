use rand::distributions::{Distribution, Uniform};
pub fn get_samples(length: usize, binsize: usize) -> Vec<usize> {
    let length_new = length / binsize;
    let mut rng = rand::thread_rng();
    let samples: Vec<_> = Uniform::from(0..length)
        .sample_iter(&mut rng)
        .take(length_new)
        .collect();
    samples
}
