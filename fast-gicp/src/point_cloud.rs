//! Point cloud types and operations.

use crate::{Error, Result};
use cxx::UniquePtr;
use fast_gicp_sys::ffi::{self};

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
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyz(),
        })
    }

    /// Creates a point cloud from a slice of points.
    pub fn from_points(points: &[[f32; 3]]) -> Result<Self> {
        let mut cloud = Self::new()?;
        cloud.reserve(points.len());

        for &[x, y, z] in points {
            cloud.push_point(x, y, z)?;
        }

        Ok(cloud)
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
    pub fn push(&mut self, point: [f32; 3]) -> Result<()> {
        self.push_point(point[0], point[1], point[2])
    }

    /// Adds a point to the cloud by coordinates.
    pub fn push_point(&mut self, x: f32, y: f32, z: f32) -> Result<()> {
        ffi::point_cloud_xyz_push_point(self.inner.pin_mut(), x, y, z);
        Ok(())
    }

    /// Gets a point from the cloud by index.
    pub fn get(&self, index: usize) -> Result<[f32; 3]> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index });
        }

        let point = ffi::point_cloud_xyz_get_point(&self.inner, index);
        Ok([point.x, point.y, point.z])
    }

    /// Sets a point in the cloud by index.
    pub fn set(&mut self, index: usize, point: [f32; 3]) -> Result<()> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index });
        }
        ffi::point_cloud_xyz_set_point(self.inner.pin_mut(), index, point[0], point[1], point[2]);
        Ok(())
    }

    /// Converts the point cloud to a vector of points.
    pub fn to_points(&self) -> Vec<[f32; 3]> {
        (0..self.size())
            .map(|i| self.get(i).expect("Index should be valid"))
            .collect()
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
        Self::new().expect("Failed to create default point cloud")
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
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ffi::create_point_cloud_xyzi(),
        })
    }

    /// Creates a point cloud from a slice of points with intensity.
    pub fn from_points(points: &[[f32; 4]]) -> Result<Self> {
        let mut cloud = Self::new()?;
        cloud.reserve(points.len());

        for &[x, y, z, intensity] in points {
            cloud.push_point(x, y, z, intensity)?;
        }

        Ok(cloud)
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
    pub fn push(&mut self, point: [f32; 4]) -> Result<()> {
        self.push_point(point[0], point[1], point[2], point[3])
    }

    /// Adds a point to the cloud by coordinates and intensity.
    pub fn push_point(&mut self, x: f32, y: f32, z: f32, intensity: f32) -> Result<()> {
        ffi::point_cloud_xyzi_push_point(self.inner.pin_mut(), x, y, z, intensity);
        Ok(())
    }

    /// Gets a point from the cloud by index.
    pub fn get(&self, index: usize) -> Result<[f32; 4]> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index });
        }

        let point = ffi::point_cloud_xyzi_get_point(&self.inner, index);
        Ok([point.x, point.y, point.z, point.intensity])
    }

    /// Sets a point in the cloud by index.
    pub fn set(&mut self, index: usize, point: [f32; 4]) -> Result<()> {
        if index >= self.size() {
            return Err(Error::IndexOutOfBounds { index });
        }
        ffi::point_cloud_xyzi_set_point(
            self.inner.pin_mut(),
            index,
            point[0],
            point[1],
            point[2],
            point[3],
        );
        Ok(())
    }

    /// Converts the point cloud to a vector of points.
    pub fn to_points(&self) -> Vec<[f32; 4]> {
        (0..self.size())
            .map(|i| self.get(i).expect("Index should be valid"))
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
        Self::new().expect("Failed to create default point cloud")
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
        let mut cloud = PointCloudXYZ::new().unwrap();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());

        cloud.push([1.0, 2.0, 3.0]).unwrap();
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
        let mut cloud = PointCloudXYZI::new().unwrap();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());

        cloud.push([1.0, 2.0, 3.0, 0.5]).unwrap();
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
