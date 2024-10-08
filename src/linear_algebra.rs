use crate::types::Triangle;
use nalgebra::{Matrix3, Point3, Rotation3, Vector3, LU};

// Check this article for the math:
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
fn intersects_triangle(
    ray_origin: &Point3<f32>,
    ray_direction: &Vector3<f32>,
    a: &Point3<f32>,
    b: &Point3<f32>,
    c: &Point3<f32>,
) -> Option<f32> {
    let a_to_b = *b - *a;
    let a_to_c = *c - *a;
    let matrix_target = ray_origin - *a;
    let intersection_matrix = Matrix3::from_columns(&[(-1.0) * ray_direction, a_to_b, a_to_c]);
    let lu = LU::new(intersection_matrix);
    let intersection = lu.solve(&matrix_target)?;
    if intersection[1] < 0.0 || intersection[1] > 1.0 {
        return None;
    }
    if intersection[2] < 0.0 || intersection[1] + intersection[2] > 1.0 {
        return None;
    }
    if intersection[0] > 1.0 || intersection[0] <= 0.0 {
        return None;
    }
    Some(intersection[0])
}

#[allow(dead_code)]
fn moller_trumbore_intersection(
    origin: &Point3<f32>,
    direction: &Vector3<f32>,
    // triangle: Triangle,
    a: &Point3<f32>,
    b: &Point3<f32>,
    c: &Point3<f32>,
) -> Option<f32> {
    let e1 = b - a;
    let e2 = c - a;

    let ray_cross_e2 = direction.cross(&e2);
    let det = e1.dot(&ray_cross_e2);

    if det > -f32::EPSILON && det < f32::EPSILON {
        return None; // This ray is parallel to this triangle.
    }

    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * s.dot(&ray_cross_e2);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let s_cross_e1 = s.cross(&e1);
    let v = inv_det * direction.dot(&s_cross_e1);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    let t = inv_det * e2.dot(&s_cross_e1);

    if t > f32::EPSILON && t <= 0.02 {
        // ray intersection
        // let intersection_point = origin + direction * t;
        // return Some(intersection_point);
        // return Some(t);
        return None;
    } else {
        // This means that there is a line intersection but not a ray intersection.
        return None;
    }
}

pub fn intersects_chip_walls(
    triangles: &Vec<Triangle>,
    ray_origin: &Point3<f32>,
    ray_direction: &Vector3<f32>,
) -> Option<(f32, Triangle)> {
    let mut intersections = std::vec::Vec::new();
    for triangle in triangles.iter() {
        let intersect = intersects_triangle( //compute the intersection of a ray and a triangle in 3d
            // let intersect = moller_trumbore_intersection(
            ray_origin,
            ray_direction,
            &triangle.a,
            &triangle.b,
            &triangle.c,
        );
        match intersect {
            Some(distance) => intersections.push((distance, triangle.clone())), //if it contains a value, namley there is an intersection
            None => continue, //next iteration (triangle)
        }
    }
    // intersections
    //     .into_iter()
    //     .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    //     .map(|(dist, tri)| (dist, tri))
    if intersections.is_empty() {
        return None;
    } else {
        return Some((1.0, triangles[0].clone()));
    }
}

// This is necessary to reflect diussing particles off wall.
#[allow(dead_code)]
pub fn reflect_on_wall(
    ray_origin: &Point3<f32>,
    ray_direction: &Vector3<f32>,
    // a: &Point3<f32>,
    // b: &Point3<f32>,
    // c: &Point3<f32>,
    triangle: &Triangle,
    step_size: &f32,
) -> (Point3<f32>, Vector3<f32>) {
    let (a, b, c) = (triangle.a, triangle.b, triangle.c);
    let a_to_b = b - a;
    let a_to_c = c - a;
    let triangle_normal_vector = a_to_b.cross(&a_to_c);
    let rotation_matrix = make_rotation(&triangle_normal_vector, std::f32::consts::PI);
    let new_direction = rotation_matrix * ray_direction * (-1.0);
    let ray_origin = ray_origin + *step_size * ray_direction;
    // Return the particle as it is reflected off the wall.
    // Need to check if it needs to be bounced again!
    return (ray_origin, (1.0 - *step_size) * new_direction);
}

pub fn make_rotation(m1: &Vector3<f32>, angle: f32) -> Rotation3<f32> {
    let rotation_matrix = Rotation3::new(m1.normalize() * angle);
    rotation_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nmr_and_mesh::make_all_triangles;
    use nalgebra::{Point3, Vector3};
    fn setup() -> (Point3<f32>, Point3<f32>, Point3<f32>, Point3<f32>) {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(0.0, 1.0, 0.0);
        let c = Point3::new(0.0, 0.0, 1.0);
        let ray_origin = Point3::new(1.0, 0.5, 0.5);
        (a, b, c, ray_origin)
    }

    #[test]
    fn no_intersection() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(2.0, 0.0, 0.0);
        assert_eq!(
            intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c),
            None
        );
    }
    #[test]
    fn intersect() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(-2.0, 0.0, 0.0);
        assert_eq!(
            intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c),
            Some(0.5)
        );
    }
    #[test]
    fn no_intersect_close() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(-0.99, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    #[test]
    fn intersect_exact() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(-1.0, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, Some(1.0));
    }
    #[test]
    fn intersect_close() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(-0.9999, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    #[test]
    fn no_intersect_outside_triangle() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(1.0, 1.5, 1.5);
        let ray_direction = Vector3::new(-2.0, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    #[test]
    fn no_intersect_unsolvable() {
        let (a, b, c, ray_origin) = setup();
        let ray_direction = Vector3::new(0.0, 1.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    #[test]
    fn intersect_closeup() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(0.000000001, 0.5, 0.5);
        let ray_direction = Vector3::new(-0.000000001, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, Some(1.0));
    }
    #[test]
    fn no_intersect_closeup() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(0.000000001, 0.5, 0.5);
        let ray_direction = Vector3::new(-0.0000000009, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    #[test]
    fn intersect_on_corner() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(1.0, 0.0, 0.0);
        let ray_direction = Vector3::new(-1.0, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, Some(1.0));
    }
    #[test]
    fn intersect_on_lower_edge() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(1.0, 0.5, 0.0);
        let ray_direction = Vector3::new(-1.0, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, Some(1.0));
    }
    #[test]
    fn no_intersect_on_lower_edge() {
        let (a, b, c, _ray_origin) = setup();
        let ray_origin = Point3::new(1.0, 0.5, 0.0 - f32::EPSILON);
        let ray_direction = Vector3::new(-1.0, 0.0, 0.0);
        let intersect_result = intersects_triangle(&ray_origin, &ray_direction, &a, &b, &c);
        assert_eq!(intersect_result, None);
    }
    // Tests on STL files
    fn set_up_triangles() -> Vec<Triangle> {
        let triangles = make_all_triangles("HollowChip.stl".into());
        dbg!(triangles.len());
        triangles
    }

    #[test]
    fn test_escape_triangle_z() {
        let triangles = set_up_triangles();
        let ray_origin = Point3::new(0.5, 0.5, 0.1);
        let ray_direction = Vector3::new(-0.0, -0.0, 1.0);
        let intersect_result = intersects_chip_walls(&triangles, &ray_origin, &ray_direction);
        assert!(intersect_result.is_some());
    }
    #[test]
    fn test_escape_triangle_z_neg() {
        let triangles = set_up_triangles();
        let ray_origin = Point3::new(0.5, 0.5, 0.1);
        let ray_direction = Vector3::new(0.0, 0.0, -1.0);
        let intersect_result = intersects_chip_walls(&triangles, &ray_origin, &ray_direction);
        assert!(intersect_result.is_some());
    }
    #[test]
    fn test_escape_triangle_y() {
        let triangles = set_up_triangles();
        let ray_origin = Point3::new(0.5, 0.5, 0.1);
        let ray_direction = Vector3::new(-0.0, 2.0, 0.0);
        let intersect_result = intersects_chip_walls(&triangles, &ray_origin, &ray_direction);
        assert!(intersect_result.is_some());
    }
    #[test]
    fn test_escape_triangle_x() {
        let triangles = set_up_triangles();
        let ray_origin = Point3::new(0.5, 0.5, 0.1);
        let ray_direction = Vector3::new(2.0, 0.0, 0.0);
        let intersect_result = intersects_chip_walls(&triangles, &ray_origin, &ray_direction);
        assert!(intersect_result.is_some());
    }
}
