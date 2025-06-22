//! Point cloud types and operations.

use crate::{FastGicpError, Result};
use cxx::UniquePtr;
use fast_gicp_sys::ffi::{self, Point3f, Point4f};
use std::pin::Pin;

/// A point cloud containing XYZ coordinates.
pub struct PointCloudXYZ {
    inner: UniquePtr<ffi::PointCloudXYZ>,
}

/// A point cloud containing XYZ coordinates and intensity values.
pub struct PointCloudXYZI {
    inner: UniquePtr<ffi::PointCloudXYZI>,
}

impl PointCloudXYZ {
    /// Creates a new empty point cloud.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_point_cloud_xyz(),
        }
    }

    /// Creates a point cloud from a slice of points.
    pub fn from_points(points: &[[f32; 3]]) -> Result<Self> {
        let ffi_points: Vec<Point3f> = points
            .iter()
            .map(|p| Point3f::new(p[0], p[1], p[2]))
            .collect();

        Ok(Self {
            inner: ffi::point_cloud_xyz_from_points(&ffi_points),
        })
    }

    /// Returns the number of points in the cloud.
    pub fn size(&self) -> usize {
        ffi::point_cloud_xyz_size(&self.inner)
    }

    /// Returns true if the cloud is empty.
    pub fn is_empty(&self) -> bool {
        ffi::point_cloud_xyz_empty(&self.inner)
    }

    /// Clears all points from the cloud.
    pub fn clear(&mut self) {
        ffi::point_cloud_xyz_clear(self.inner.pin_mut());
    }

    /// Reserves space for the specified number of points.
    pub fn reserve(&mut self, size: usize) {
        ffi::point_cloud_xyz_reserve(self.inner.pin_mut(), size);
    }

    /// Adds a point to the cloud.
    pub fn push(&mut self, point: [f32; 3]) {
        let ffi_point = Point3f::new(point[0], point[1], point[2]);
        ffi::point_cloud_xyz_push_back(self.inner.pin_mut(), &ffi_point);
    }

    /// Gets a point from the cloud by index.
    pub fn get(&self, index: usize) -> Result<[f32; 3]> {
        if index >= self.size() {
            return Err(FastGicpError::IndexOutOfBounds { index });
        }

        let point = ffi::point_cloud_xyz_get_point(&self.inner, index);
        Ok([point.x, point.y, point.z])
    }

    /// Converts the point cloud to a vector of points.
    pub fn to_points(&self) -> Vec<[f32; 3]> {
        let ffi_points = ffi::point_cloud_xyz_to_points(&self.inner);
        ffi_points.into_iter().map(|p| [p.x, p.y, p.z]).collect()
    }

    /// Returns an iterator over the points in the cloud.
    pub fn iter(&self) -> PointCloudXYZIter {
        PointCloudXYZIter {
            cloud: self,
            index: 0,
        }
    }

    /// Internal method to get access to the underlying C++ object.
    pub(crate) fn as_ffi(&self) -> &ffi::PointCloudXYZ {
        &self.inner
    }
}

impl Default for PointCloudXYZ {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<[f32; 3]> for PointCloudXYZ {
    fn from_iter<T: IntoIterator<Item = [f32; 3]>>(iter: T) -> Self {
        let points: Vec<[f32; 3]> = iter.into_iter().collect();
        Self::from_points(&points).expect("Failed to create point cloud")
    }
}

impl FromIterator<(f32, f32, f32)> for PointCloudXYZ {
    fn from_iter<T: IntoIterator<Item = (f32, f32, f32)>>(iter: T) -> Self {
        let points: Vec<[f32; 3]> = iter.into_iter().map(|(x, y, z)| [x, y, z]).collect();
        Self::from_points(&points).expect("Failed to create point cloud")
    }
}

/// Iterator over points in a PointCloudXYZ.
pub struct PointCloudXYZIter<'a> {
    cloud: &'a PointCloudXYZ,
    index: usize,
}

impl<'a> Iterator for PointCloudXYZIter<'a> {
    type Item = [f32; 3];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.cloud.size() {
            let point = self.cloud.get(self.index).ok()?;
            self.index += 1;
            Some(point)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.cloud.size().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PointCloudXYZIter<'a> {}

impl PointCloudXYZI {
    /// Creates a new empty point cloud.
    pub fn new() -> Self {
        Self {
            inner: ffi::create_point_cloud_xyzi(),
        }
    }

    /// Creates a point cloud from a slice of points with intensity.
    pub fn from_points(points: &[[f32; 4]]) -> Result<Self> {
        let ffi_points: Vec<Point4f> = points
            .iter()
            .map(|p| Point4f::new(p[0], p[1], p[2], p[3]))
            .collect();

        Ok(Self {
            inner: ffi::point_cloud_xyzi_from_points(&ffi_points),
        })
    }

    /// Returns the number of points in the cloud.
    pub fn size(&self) -> usize {
        ffi::point_cloud_xyzi_size(&self.inner)
    }

    /// Returns true if the cloud is empty.
    pub fn is_empty(&self) -> bool {
        ffi::point_cloud_xyzi_empty(&self.inner)
    }

    /// Clears all points from the cloud.
    pub fn clear(&mut self) {
        ffi::point_cloud_xyzi_clear(self.inner.pin_mut());
    }

    /// Reserves space for the specified number of points.
    pub fn reserve(&mut self, size: usize) {
        ffi::point_cloud_xyzi_reserve(self.inner.pin_mut(), size);
    }

    /// Adds a point to the cloud.
    pub fn push(&mut self, point: [f32; 4]) {
        let ffi_point = Point4f::new(point[0], point[1], point[2], point[3]);
        ffi::point_cloud_xyzi_push_back(self.inner.pin_mut(), &ffi_point);
    }

    /// Gets a point from the cloud by index.
    pub fn get(&self, index: usize) -> Result<[f32; 4]> {
        if index >= self.size() {
            return Err(FastGicpError::IndexOutOfBounds { index });
        }

        let point = ffi::point_cloud_xyzi_get_point(&self.inner, index);
        Ok([point.x, point.y, point.z, point.intensity])
    }

    /// Converts the point cloud to a vector of points.
    pub fn to_points(&self) -> Vec<[f32; 4]> {
        let ffi_points = ffi::point_cloud_xyzi_to_points(&self.inner);
        ffi_points
            .into_iter()
            .map(|p| [p.x, p.y, p.z, p.intensity])
            .collect()
    }

    /// Returns an iterator over the points in the cloud.
    pub fn iter(&self) -> PointCloudXYZIIter {
        PointCloudXYZIIter {
            cloud: self,
            index: 0,
        }
    }

    /// Internal method to get access to the underlying C++ object.
    pub(crate) fn as_ffi(&self) -> &ffi::PointCloudXYZI {
        &self.inner
    }
}

impl Default for PointCloudXYZI {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<[f32; 4]> for PointCloudXYZI {
    fn from_iter<T: IntoIterator<Item = [f32; 4]>>(iter: T) -> Self {
        let points: Vec<[f32; 4]> = iter.into_iter().collect();
        Self::from_points(&points).expect("Failed to create point cloud")
    }
}

impl FromIterator<(f32, f32, f32, f32)> for PointCloudXYZI {
    fn from_iter<T: IntoIterator<Item = (f32, f32, f32, f32)>>(iter: T) -> Self {
        let points: Vec<[f32; 4]> = iter.into_iter().map(|(x, y, z, i)| [x, y, z, i]).collect();
        Self::from_points(&points).expect("Failed to create point cloud")
    }
}

/// Iterator over points in a PointCloudXYZI.
pub struct PointCloudXYZIIter<'a> {
    cloud: &'a PointCloudXYZI,
    index: usize,
}

impl<'a> Iterator for PointCloudXYZIIter<'a> {
    type Item = [f32; 4];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.cloud.size() {
            let point = self.cloud.get(self.index).ok()?;
            self.index += 1;
            Some(point)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.cloud.size().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PointCloudXYZIIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_xyz_basic_operations() {
        let mut cloud = PointCloudXYZ::new();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());

        cloud.push([1.0, 2.0, 3.0]);
        assert_eq!(cloud.size(), 1);
        assert!(!cloud.is_empty());

        let point = cloud.get(0).unwrap();
        assert_eq!(point, [1.0, 2.0, 3.0]);

        cloud.clear();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());
    }

    #[test]
    fn test_point_cloud_xyz_from_iterator() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let cloud: PointCloudXYZ = points.iter().copied().collect();

        assert_eq!(cloud.size(), 3);
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(cloud.get(1).unwrap(), [4.0, 5.0, 6.0]);
        assert_eq!(cloud.get(2).unwrap(), [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_point_cloud_xyz_iterator() {
        let points = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cloud: PointCloudXYZ = points.iter().copied().collect();

        let collected: Vec<[f32; 3]> = cloud.iter().collect();
        assert_eq!(collected, points);
    }

    #[test]
    fn test_point_cloud_xyzi_basic_operations() {
        let mut cloud = PointCloudXYZI::new();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());

        cloud.push([1.0, 2.0, 3.0, 0.5]);
        assert_eq!(cloud.size(), 1);
        assert!(!cloud.is_empty());

        let point = cloud.get(0).unwrap();
        assert_eq!(point, [1.0, 2.0, 3.0, 0.5]);
    }

    #[test]
    fn test_point_cloud_xyzi_from_iterator() {
        let points = vec![[1.0, 2.0, 3.0, 0.1], [4.0, 5.0, 6.0, 0.2]];
        let cloud: PointCloudXYZI = points.iter().copied().collect();

        assert_eq!(cloud.size(), 2);
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0, 0.1]);
        assert_eq!(cloud.get(1).unwrap(), [4.0, 5.0, 6.0, 0.2]);
    }
}
