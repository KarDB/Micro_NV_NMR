mod nmr_basic;
use bvh::Point3;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// parallelization benchmark
//pub fn criterion_benchmark(c: &mut criterion) {
//    let mut rng = smallrng::from_entropy();
//    let m1 = point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
//    let m2 = point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
//    let mut pos = nmr::make_nv_locations(0.1);
//    c.bench_function("parallel protons", |b| {
//        b.iter(|| nmr::start(black_box(m1), black_box(m2), black_box(&mut pos)))
//    });
//}
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    let m1 = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let r = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let m2 = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let mut pos = nmr_basic::make_nv_locations(0.1);
    c.bench_function("dd optimization", |b| {
        b.iter(|| {
            nmr_basic::dd_for_all_pos(
                black_box(&r),
                black_box(&m1),
                black_box(&m2),
                black_box(&mut pos),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
