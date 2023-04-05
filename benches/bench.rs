mod nmr;
use bvh::Point3;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = SmallRng::from_entropy();
    let r = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let m1 = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let m2 = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
    let mut pos = nmr::make_nv_locations(0.1);
    c.bench_function("parallel protons", |b| {
        b.iter(|| {
            nmr::dd_for_all_pos(
                black_box(r),
                black_box(m1),
                black_box(m2),
                black_box(&mut pos),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
