//! 3D transformation utilities.

use fast_gicp_sys::ffi;
use nalgebra::{Isometry3, Matrix4, UnitQuaternion, Vector3};

/// A 3D transformation represented as a 4x4 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform3f {
    inner: ffi::Transform3f,
}

impl Transform3f {
    /// Creates an identity transformation.
    pub fn identity() -> Self {
        Self {
            inner: ffi::Transform3f::identity(),
        }
    }

    /// Creates a transformation from a translation vector.
    pub fn from_translation(translation: [f32; 3]) -> Self {
        let mut matrix = [0.0f32; 16];
        // Identity matrix
        matrix[0] = 1.0; // m00
        matrix[5] = 1.0; // m11
        matrix[10] = 1.0; // m22
        matrix[15] = 1.0; // m33

        // Translation
        matrix[12] = translation[0]; // m03
        matrix[13] = translation[1]; // m13
        matrix[14] = translation[2]; // m23

        Self {
            inner: ffi::Transform3f { matrix },
        }
    }

    /// Creates a transformation from a rotation matrix (3x3) and translation.
    pub fn from_rotation_translation(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        let mut matrix = [0.0f32; 16];

        // Set rotation part (column-major)
        matrix[0] = rotation[0][0]; // m00
        matrix[1] = rotation[1][0]; // m10
        matrix[2] = rotation[2][0]; // m20
        matrix[3] = 0.0; // m30

        matrix[4] = rotation[0][1]; // m01
        matrix[5] = rotation[1][1]; // m11
        matrix[6] = rotation[2][1]; // m21
        matrix[7] = 0.0; // m31

        matrix[8] = rotation[0][2]; // m02
        matrix[9] = rotation[1][2]; // m12
        matrix[10] = rotation[2][2]; // m22
        matrix[11] = 0.0; // m32

        // Set translation part
        matrix[12] = translation[0]; // m03
        matrix[13] = translation[1]; // m13
        matrix[14] = translation[2]; // m23
        matrix[15] = 1.0; // m33

        Self {
            inner: ffi::Transform3f { matrix },
        }
    }

    /// Creates a transformation from a nalgebra Isometry3.
    pub fn from_isometry(isometry: &Isometry3<f32>) -> Self {
        let matrix = isometry.to_homogeneous();
        let mut array = [0.0f32; 16];

        // Convert column-major nalgebra matrix to our format
        for col in 0..4 {
            for row in 0..4 {
                array[col * 4 + row] = matrix[(row, col)];
            }
        }

        Self {
            inner: ffi::Transform3f { matrix: array },
        }
    }

    /// Converts to a nalgebra Isometry3.
    pub fn to_isometry(&self) -> Isometry3<f32> {
        let mut matrix = Matrix4::<f32>::zeros();

        // Convert our format to column-major nalgebra matrix
        for col in 0..4 {
            for row in 0..4 {
                matrix[(row, col)] = self.inner.matrix[col * 4 + row];
            }
        }

        Isometry3::from_matrix_unchecked(matrix)
    }

    /// Gets the translation component as a vector.
    pub fn translation(&self) -> [f32; 3] {
        [
            self.inner.matrix[12], // m03
            self.inner.matrix[13], // m13
            self.inner.matrix[14], // m23
        ]
    }

    /// Gets the rotation component as a 3x3 matrix.
    pub fn rotation(&self) -> [[f32; 3]; 3] {
        [
            [
                self.inner.matrix[0],
                self.inner.matrix[4],
                self.inner.matrix[8],
            ], // row 0
            [
                self.inner.matrix[1],
                self.inner.matrix[5],
                self.inner.matrix[9],
            ], // row 1
            [
                self.inner.matrix[2],
                self.inner.matrix[6],
                self.inner.matrix[10],
            ], // row 2
        ]
    }

    /// Gets the raw 4x4 matrix in column-major order.
    pub fn matrix(&self) -> &[f32; 16] {
        &self.inner.matrix
    }

    /// Composes this transformation with another (self * other).
    pub fn compose(&self, other: &Transform3f) -> Transform3f {
        let iso1 = self.to_isometry();
        let iso2 = other.to_isometry();
        let result = iso1 * iso2;
        Transform3f::from_isometry(&result)
    }

    /// Inverts the transformation.
    pub fn inverse(&self) -> Transform3f {
        let iso = self.to_isometry();
        let inv = iso.inverse();
        Transform3f::from_isometry(&inv)
    }

    /// Internal method to get access to the underlying FFI type.
    pub(crate) fn as_ffi(&self) -> &ffi::Transform3f {
        &self.inner
    }

    /// Internal method to create from FFI type.
    pub(crate) fn from_ffi(transform: ffi::Transform3f) -> Self {
        Self { inner: transform }
    }
}

impl Default for Transform3f {
    fn default() -> Self {
        Self::identity()
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
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_identity_transform() {
        let transform = Transform3f::identity();
        let translation = transform.translation();
        assert_eq!(translation, [0.0, 0.0, 0.0]);

        let rotation = transform.rotation();
        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(rotation, expected_rotation);
    }

    #[test]
    fn test_translation_transform() {
        let translation = [1.0, 2.0, 3.0];
        let transform = Transform3f::from_translation(translation);
        assert_eq!(transform.translation(), translation);

        let rotation = transform.rotation();
        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(rotation, expected_rotation);
    }

    #[test]
    fn test_nalgebra_conversion() {
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
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
        let t1 = Transform3f::from_translation([1.0, 0.0, 0.0]);
        let t2 = Transform3f::from_translation([0.0, 1.0, 0.0]);
        let composed = t1.compose(&t2);

        let expected_translation = [1.0, 1.0, 0.0];
        let actual_translation = composed.translation();

        for i in 0..3 {
            assert!((actual_translation[i] - expected_translation[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_transform_inverse() {
        let translation = [1.0, 2.0, 3.0];
        let transform = Transform3f::from_translation(translation);
        let inverse = transform.inverse();
        let identity = transform.compose(&inverse);

        let result_translation = identity.translation();
        for i in 0..3 {
            assert!(
                result_translation[i].abs() < 1e-6,
                "Translation component {} is {}",
                i,
                result_translation[i]
            );
        }
    }
}
