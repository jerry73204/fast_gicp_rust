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
        // PCL internally handles memory allocation, so we don't need explicit capacity checks
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

    /// Extends the point cloud with points from an iterator.
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = [f32; 3]>,
    {
        for point in iter {
            self.push(point)?;
        }
        Ok(())
    }

    /// Appends all points from another point cloud.
    pub fn append(&mut self, other: &PointCloudXYZ) -> Result<()> {
        self.extend(other.iter())
    }

    /// Transforms all points in the cloud by the given transformation.
    pub fn transform(&mut self, transform: &crate::Transform3f) -> Result<()> {
        for i in 0..self.size() {
            let point = self.get(i)?;
            let transformed = transform.transform_point_array(point);
            self.set(i, transformed)?;
        }
        Ok(())
    }

    /// Creates a new transformed point cloud without modifying the original.
    pub fn transformed(&self, transform: &crate::Transform3f) -> Result<Self> {
        let mut result = Self::new()?;
        result.reserve(self.size());

        for point in self.iter() {
            let transformed = transform.transform_point_array(point);
            result.push(transformed)?;
        }

        Ok(result)
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

impl<'a> IntoIterator for &'a PointCloudXYZ {
    type Item = [f32; 3];
    type IntoIter = PointCloudXYZIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Extend<[f32; 3]> for PointCloudXYZ {
    fn extend<T: IntoIterator<Item = [f32; 3]>>(&mut self, iter: T) {
        for point in iter {
            // Ignore errors during extend
            let _ = self.push(point);
        }
    }
}

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

    /// Extends the point cloud with points from an iterator.
    pub fn extend<I>(&mut self, iter: I) -> Result<()>
    where
        I: IntoIterator<Item = [f32; 4]>,
    {
        for point in iter {
            self.push(point)?;
        }
        Ok(())
    }

    /// Appends all points from another point cloud.
    pub fn append(&mut self, other: &PointCloudXYZI) -> Result<()> {
        self.extend(other.iter())
    }

    /// Transforms all points in the cloud by the given transformation.
    /// The intensity values are preserved.
    pub fn transform(&mut self, transform: &crate::Transform3f) -> Result<()> {
        for i in 0..self.size() {
            let point = self.get(i)?;
            let xyz = [point[0], point[1], point[2]];
            let transformed_xyz = transform.transform_point_array(xyz);
            self.set(
                i,
                [
                    transformed_xyz[0],
                    transformed_xyz[1],
                    transformed_xyz[2],
                    point[3],
                ],
            )?;
        }
        Ok(())
    }

    /// Creates a new transformed point cloud without modifying the original.
    /// The intensity values are preserved.
    pub fn transformed(&self, transform: &crate::Transform3f) -> Result<Self> {
        let mut result = Self::new()?;
        result.reserve(self.size());

        for point in self.iter() {
            let xyz = [point[0], point[1], point[2]];
            let transformed_xyz = transform.transform_point_array(xyz);
            result.push([
                transformed_xyz[0],
                transformed_xyz[1],
                transformed_xyz[2],
                point[3],
            ])?;
        }

        Ok(result)
    }

    /// Returns an iterator over the points in the cloud.
    pub fn iter(&self) -> PointCloudXYZIIter {
        PointCloudXYZIIter {
            cloud: self,
            index: 0,
        }
    }

    /// Internal method to get access to the underlying C++ object.
    #[allow(dead_code)]
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

impl<'a> IntoIterator for &'a PointCloudXYZI {
    type Item = [f32; 4];
    type IntoIter = PointCloudXYZIIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Extend<[f32; 4]> for PointCloudXYZI {
    fn extend<T: IntoIterator<Item = [f32; 4]>>(&mut self, iter: T) {
        for point in iter {
            // Ignore errors during extend
            let _ = self.push(point);
        }
    }
}

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
        let points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let cloud: PointCloudXYZ = points.iter().copied().collect();

        assert_eq!(cloud.size(), 3);
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(cloud.get(1).unwrap(), [4.0, 5.0, 6.0]);
        assert_eq!(cloud.get(2).unwrap(), [7.0, 8.0, 9.0]);
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
        let points = [[1.0, 2.0, 3.0, 0.1], [4.0, 5.0, 6.0, 0.2]];
        let cloud: PointCloudXYZI = points.iter().copied().collect();

        assert_eq!(cloud.size(), 2);
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0, 0.1]);
        assert_eq!(cloud.get(1).unwrap(), [4.0, 5.0, 6.0, 0.2]);
    }

    #[test]
    fn test_point_cloud_xyz_iterator() {
        let points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let cloud: PointCloudXYZ = points.iter().copied().collect();

        // Test iter() method
        let collected: Vec<[f32; 3]> = cloud.iter().collect();
        assert_eq!(collected, points);

        // Test IntoIterator trait
        let mut count = 0;
        for point in &cloud {
            assert_eq!(point, points[count]);
            count += 1;
        }
        assert_eq!(count, 3);

        // Test ExactSizeIterator
        assert_eq!(cloud.iter().len(), 3);
    }

    #[test]
    fn test_point_cloud_xyzi_iterator() {
        let points = [[1.0, 2.0, 3.0, 0.1], [4.0, 5.0, 6.0, 0.2]];
        let cloud: PointCloudXYZI = points.iter().copied().collect();

        // Test iter() method
        let collected: Vec<[f32; 4]> = cloud.iter().collect();
        assert_eq!(collected, points);

        // Test IntoIterator trait
        let mut count = 0;
        for point in &cloud {
            assert_eq!(point, points[count]);
            count += 1;
        }
        assert_eq!(count, 2);

        // Test ExactSizeIterator
        assert_eq!(cloud.iter().len(), 2);
    }

    #[test]
    fn test_bounds_checking() {
        let mut cloud = PointCloudXYZ::new().unwrap();
        cloud.push([1.0, 2.0, 3.0]).unwrap();
        cloud.push([4.0, 5.0, 6.0]).unwrap();

        // Valid access
        assert!(cloud.get(0).is_ok());
        assert!(cloud.get(1).is_ok());

        // Out of bounds access
        match cloud.get(2) {
            Err(Error::IndexOutOfBounds { index }) => assert_eq!(index, 2),
            _ => panic!("Expected IndexOutOfBounds error"),
        }

        // Out of bounds set
        match cloud.set(2, [7.0, 8.0, 9.0]) {
            Err(Error::IndexOutOfBounds { index }) => assert_eq!(index, 2),
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }

    #[test]
    fn test_clear_and_reserve() {
        let mut cloud = PointCloudXYZ::new().unwrap();

        // Add some points
        cloud.push([1.0, 2.0, 3.0]).unwrap();
        cloud.push([4.0, 5.0, 6.0]).unwrap();
        assert_eq!(cloud.size(), 2);

        // Clear the cloud
        cloud.clear();
        assert_eq!(cloud.size(), 0);
        assert!(cloud.is_empty());

        // Reserve capacity
        cloud.reserve(100);
        assert!(cloud.is_empty()); // Still empty after reserve

        // Can still add points after clear
        cloud.push([7.0, 8.0, 9.0]).unwrap();
        assert_eq!(cloud.size(), 1);
    }

    #[test]
    fn test_batch_operations() {
        let mut cloud = PointCloudXYZ::new().unwrap();

        // Test extend with iterator
        let points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        cloud.extend(points.iter().copied()).unwrap();
        assert_eq!(cloud.size(), 3);

        // Test Extend trait
        let more_points = [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]];
        cloud.extend(more_points.iter().copied()).unwrap();
        assert_eq!(cloud.size(), 5);

        // Test append
        let mut other_cloud = PointCloudXYZ::new().unwrap();
        other_cloud.push([16.0, 17.0, 18.0]).unwrap();
        other_cloud.push([19.0, 20.0, 21.0]).unwrap();

        cloud.append(&other_cloud).unwrap();
        assert_eq!(cloud.size(), 7);

        // Verify all points
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(cloud.get(6).unwrap(), [19.0, 20.0, 21.0]);
    }

    #[test]
    fn test_batch_operations_xyzi() {
        let mut cloud = PointCloudXYZI::new().unwrap();

        // Test extend with iterator
        let points = [[1.0, 2.0, 3.0, 0.1], [4.0, 5.0, 6.0, 0.2]];
        cloud.extend(points.iter().copied()).unwrap();
        assert_eq!(cloud.size(), 2);

        // Test Extend trait
        cloud
            .extend([[7.0, 8.0, 9.0, 0.3]].iter().copied())
            .unwrap();
        assert_eq!(cloud.size(), 3);

        // Test append
        let mut other_cloud = PointCloudXYZI::new().unwrap();
        other_cloud.push([10.0, 11.0, 12.0, 0.4]).unwrap();

        cloud.append(&other_cloud).unwrap();
        assert_eq!(cloud.size(), 4);

        // Verify points
        assert_eq!(cloud.get(0).unwrap(), [1.0, 2.0, 3.0, 0.1]);
        assert_eq!(cloud.get(3).unwrap(), [10.0, 11.0, 12.0, 0.4]);
    }

    #[test]
    fn test_point_cloud_transformation() {
        let mut cloud = PointCloudXYZ::new().unwrap();
        cloud.push([1.0, 0.0, 0.0]).unwrap();
        cloud.push([0.0, 1.0, 0.0]).unwrap();
        cloud.push([0.0, 0.0, 1.0]).unwrap();

        // Test translation
        let translation = crate::Transform3f::from_translation(1.0, 2.0, 3.0);
        let transformed = cloud.transformed(&translation).unwrap();

        assert_eq!(transformed.get(0).unwrap(), [2.0, 2.0, 3.0]);
        assert_eq!(transformed.get(1).unwrap(), [1.0, 3.0, 3.0]);
        assert_eq!(transformed.get(2).unwrap(), [1.0, 2.0, 4.0]);

        // Test in-place transformation
        cloud.transform(&translation).unwrap();
        assert_eq!(cloud.get(0).unwrap(), [2.0, 2.0, 3.0]);
        assert_eq!(cloud.get(1).unwrap(), [1.0, 3.0, 3.0]);
        assert_eq!(cloud.get(2).unwrap(), [1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_point_cloud_xyzi_transformation() {
        let mut cloud = PointCloudXYZI::new().unwrap();
        cloud.push([1.0, 0.0, 0.0, 0.5]).unwrap();
        cloud.push([0.0, 1.0, 0.0, 0.7]).unwrap();

        // Test that intensity is preserved during transformation
        let translation = crate::Transform3f::from_translation(10.0, 20.0, 30.0);
        let transformed = cloud.transformed(&translation).unwrap();

        assert_eq!(transformed.get(0).unwrap(), [11.0, 20.0, 30.0, 0.5]);
        assert_eq!(transformed.get(1).unwrap(), [10.0, 21.0, 30.0, 0.7]);

        // Verify intensity values are preserved
        assert_eq!(transformed.get(0).unwrap()[3], 0.5);
        assert_eq!(transformed.get(1).unwrap()[3], 0.7);
    }
}
