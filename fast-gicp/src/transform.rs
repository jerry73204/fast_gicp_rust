//! 3D transformation utilities

use fast_gicp_sys::ffi;
use nalgebra::{Isometry3, Matrix3, Matrix4, Vector3};

/// 3D transformation matrix (4x4 homogeneous transformation)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3f {
    /// 4x4 transformation matrix in row-major order
    pub matrix: [[f32; 4]; 4],
}

impl Transform3f {
    /// Create an identity transformation
    pub fn identity() -> Self {
        let ffi_transform = ffi::transform_identity();
        Self::from_transform4f(&ffi_transform)
    }

    /// Create transformation from flat array (row-major order)
    pub fn from_flat(data: &[f32; 16]) -> Self {
        let mut matrix = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                matrix[i][j] = data[i * 4 + j];
            }
        }
        Self { matrix }
    }

    /// Convert to flat array (row-major order)
    pub fn to_flat(&self) -> [f32; 16] {
        let mut data = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 4 + j] = self.matrix[i][j];
            }
        }
        data
    }

    /// Create transformation from translation
    pub fn from_translation(x: f32, y: f32, z: f32) -> Self {
        let transform4f = ffi::transform_from_translation(x, y, z);
        Self::from_transform4f(&transform4f)
    }

    /// Creates a transformation from a translation vector.
    pub fn from_translation_array(translation: [f32; 3]) -> Self {
        Self::from_translation(translation[0], translation[1], translation[2])
    }

    /// Get translation component
    pub fn translation(&self) -> Vector3<f32> {
        Vector3::new(self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])
    }

    /// Set translation component
    pub fn set_translation(&mut self, x: f32, y: f32, z: f32) {
        self.matrix[0][3] = x;
        self.matrix[1][3] = y;
        self.matrix[2][3] = z;
    }

    /// Get rotation matrix (3x3 upper-left block)
    pub fn rotation(&self) -> Matrix3<f32> {
        Matrix3::new(
            self.matrix[0][0],
            self.matrix[0][1],
            self.matrix[0][2],
            self.matrix[1][0],
            self.matrix[1][1],
            self.matrix[1][2],
            self.matrix[2][0],
            self.matrix[2][1],
            self.matrix[2][2],
        )
    }

    /// Set rotation matrix (3x3 upper-left block)
    pub fn set_rotation(&mut self, rotation: [[f32; 3]; 3]) {
        for (i, row) in rotation.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                self.matrix[i][j] = val;
            }
        }
    }

    /// Create transformation from rotation matrix and translation
    pub fn from_rotation_translation(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        let mut transform = Self::identity();
        transform.set_rotation(rotation);
        transform.set_translation(translation[0], translation[1], translation[2]);
        transform
    }

    /// Create transformation from nalgebra translation and rotation
    pub fn from_parts(translation: Vector3<f32>, rotation: Matrix3<f32>) -> Self {
        let mut transform = Self::identity();

        // Set rotation from Matrix3
        for i in 0..3 {
            for j in 0..3 {
                transform.matrix[i][j] = rotation[(i, j)];
            }
        }

        // Set translation
        transform.matrix[0][3] = translation.x;
        transform.matrix[1][3] = translation.y;
        transform.matrix[2][3] = translation.z;

        transform
    }

    /// Create transformation from a rotation matrix
    pub fn from_rotation_matrix(rotation: &Matrix3<f32>) -> Self {
        Self::from_parts(Vector3::zeros(), *rotation)
    }

    /// Multiply two transformations
    pub fn multiply(&self, other: &Transform3f) -> Self {
        let a = self.as_transform4f();
        let b = other.as_transform4f();
        let result = ffi::transform_multiply(&a, &b);
        Self::from_transform4f(&result)
    }

    /// Composes this transformation with another (self * other).
    pub fn compose(&self, other: &Transform3f) -> Transform3f {
        self.multiply(other)
    }

    /// Compute inverse transformation
    pub fn inverse(&self) -> Self {
        let transform4f = self.as_transform4f();
        let result = ffi::transform_inverse(&transform4f);
        Self::from_transform4f(&result)
    }

    /// Transform a 3D point
    pub fn transform_point(&self, point: &Vector3<f32>) -> Vector3<f32> {
        let x = point.x;
        let y = point.y;
        let z = point.z;
        let tx = self.matrix[0][0] * x
            + self.matrix[0][1] * y
            + self.matrix[0][2] * z
            + self.matrix[0][3];
        let ty = self.matrix[1][0] * x
            + self.matrix[1][1] * y
            + self.matrix[1][2] * z
            + self.matrix[1][3];
        let tz = self.matrix[2][0] * x
            + self.matrix[2][1] * y
            + self.matrix[2][2] * z
            + self.matrix[2][3];
        Vector3::new(tx, ty, tz)
    }

    /// Transform a 3D point array
    pub fn transform_point_array(&self, point: [f32; 3]) -> [f32; 3] {
        let v = self.transform_point(&Vector3::new(point[0], point[1], point[2]));
        [v.x, v.y, v.z]
    }

    /// Create from nalgebra Isometry3
    pub fn from_isometry(isometry: &Isometry3<f32>) -> Self {
        let matrix = isometry.to_homogeneous();
        let mut flat = [0.0f32; 16];

        // Convert column-major nalgebra matrix to row-major
        for i in 0..4 {
            for j in 0..4 {
                flat[i * 4 + j] = matrix[(i, j)];
            }
        }

        Self::from_flat(&flat)
    }

    /// Convert to nalgebra Isometry3
    pub fn to_isometry(&self) -> Isometry3<f32> {
        let mut matrix = Matrix4::<f32>::zeros();

        // Convert row-major to column-major
        for i in 0..4 {
            for j in 0..4 {
                matrix[(i, j)] = self.matrix[i][j];
            }
        }

        // Extract rotation and translation from the matrix
        let rotation = matrix.fixed_view::<3, 3>(0, 0);
        let translation = matrix.fixed_view::<3, 1>(0, 3);

        Isometry3::from_parts(
            nalgebra::Translation3::from(nalgebra::Vector3::new(
                translation[(0, 0)],
                translation[(1, 0)],
                translation[(2, 0)],
            )),
            nalgebra::UnitQuaternion::from_matrix(&rotation.into()),
        )
    }

    /// Get the raw 4x4 matrix in row-major order
    pub fn matrix_data(&self) -> &[[f32; 4]; 4] {
        &self.matrix
    }

    /// Convert to nalgebra Matrix4
    pub fn to_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_fn(|i, j| self.matrix[i][j])
    }

    /// Create from nalgebra Matrix4
    pub fn from_matrix(matrix: &Matrix4<f32>) -> Self {
        let mut transform = Self::identity();
        for i in 0..4 {
            for j in 0..4 {
                transform.matrix[i][j] = matrix[(i, j)];
            }
        }
        transform
    }

    /// Convert to FFI Transform4f
    pub(crate) fn as_transform4f(&self) -> ffi::Transform4f {
        ffi::Transform4f {
            data: self.to_flat(),
        }
    }

    /// Create from FFI Transform4f
    pub(crate) fn from_transform4f(transform: &ffi::Transform4f) -> Self {
        Self::from_flat(&transform.data)
    }
}

impl Default for Transform3f {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Transform3f {
    type Output = Transform3f;

    fn mul(self, rhs: Transform3f) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl std::ops::MulAssign for Transform3f {
    fn mul_assign(&mut self, rhs: Transform3f) {
        *self = self.multiply(&rhs);
    }
}

// Conversion for nalgebra interop
impl From<nalgebra::Matrix4<f32>> for Transform3f {
    fn from(m: nalgebra::Matrix4<f32>) -> Self {
        let mut flat = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                flat[i * 4 + j] = m[(i, j)];
            }
        }
        Self::from_flat(&flat)
    }
}

impl From<Transform3f> for nalgebra::Matrix4<f32> {
    fn from(t: Transform3f) -> Self {
        let mut matrix = nalgebra::Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                matrix[(i, j)] = t.matrix[i][j];
            }
        }
        matrix
    }
}

impl From<Isometry3<f32>> for Transform3f {
    fn from(isometry: Isometry3<f32>) -> Self {
        Self::from_isometry(&isometry)
    }
}

impl From<&Transform3f> for Isometry3<f32> {
    fn from(transform: &Transform3f) -> Self {
        transform.to_isometry()
    }
}

impl From<Transform3f> for Isometry3<f32> {
    fn from(transform: Transform3f) -> Self {
        transform.to_isometry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let identity = Transform3f::identity();
        assert_eq!(identity.translation(), Vector3::new(0.0, 0.0, 0.0));

        let point = Vector3::new(1.0, 2.0, 3.0);
        let transformed = identity.transform_point(&point);
        assert_eq!(transformed, point);
    }

    #[test]
    fn test_translation() {
        let transform = Transform3f::from_translation(1.0, 2.0, 3.0);
        assert_eq!(transform.translation(), Vector3::new(1.0, 2.0, 3.0));

        let point = Vector3::new(0.0, 0.0, 0.0);
        let transformed = transform.transform_point(&point);
        assert_eq!(transformed, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_translation_array() {
        let translation = [1.0, 2.0, 3.0];
        let transform = Transform3f::from_translation_array(translation);
        assert_eq!(transform.translation(), Vector3::new(1.0, 2.0, 3.0));

        let rotation = transform.rotation();
        let expected_rotation = Matrix3::identity();
        assert_eq!(rotation, expected_rotation);
    }

    #[test]
    fn test_inverse() {
        let transform = Transform3f::from_translation(1.0, 2.0, 3.0);
        let inverse = transform.inverse();
        let identity = transform.multiply(&inverse);

        // Check if result is approximately identity
        let point = Vector3::new(5.0, 6.0, 7.0);
        let original = identity.transform_point(&point);
        assert!((original.x - point.x).abs() < 1e-6);
        assert!((original.y - point.y).abs() < 1e-6);
        assert!((original.z - point.z).abs() < 1e-6);
    }

    #[test]
    fn test_flat_conversion() {
        let identity = Transform3f::identity();
        let flat = identity.to_flat();
        let restored = Transform3f::from_flat(&flat);
        assert_eq!(identity, restored);
    }

    #[test]
    fn test_nalgebra_conversion() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let isometry = Isometry3::from_parts(translation.into(), rotation);

        let transform = Transform3f::from_isometry(&isometry);
        let back_to_isometry = transform.to_isometry();

        // Check that the round-trip preserves the transformation (within floating point precision)
        let diff = (isometry.to_homogeneous() - back_to_isometry.to_homogeneous()).abs();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    diff[(i, j)] < 1e-6,
                    "Matrix element ({}, {}) differs by {}",
                    i,
                    j,
                    diff[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_transform_composition() {
        let t1 = Transform3f::from_translation(1.0, 0.0, 0.0);
        let t2 = Transform3f::from_translation(0.0, 1.0, 0.0);
        let composed = t1.compose(&t2);

        let expected_translation = Vector3::new(1.0, 1.0, 0.0);
        let actual_translation = composed.translation();

        assert!((actual_translation - expected_translation).norm() < 1e-6);
    }

    #[test]
    fn test_transform_inverse() {
        let translation = [1.0, 2.0, 3.0];
        let transform = Transform3f::from_translation_array(translation);
        let inverse = transform.inverse();
        let identity = transform.compose(&inverse);

        let result_translation = identity.translation();
        assert!(
            result_translation.norm() < 1e-6,
            "Translation should be near zero, got {result_translation:?}"
        );
    }
}
