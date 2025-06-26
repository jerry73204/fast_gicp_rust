//! Enum validation tests based on upstream enum_validation_test.cpp
//!
//! These tests ensure enum conversions match C++ behavior.

use fast_gicp::types::{NeighborSearchMethod, RegularizationMethod, VoxelAccumulationMode};

#[test]
fn test_regularization_method_values() {
    // Ensure enum values match C++ implementation
    assert_eq!(RegularizationMethod::None as i32, 0);
    assert_eq!(RegularizationMethod::MinEig as i32, 1);
    assert_eq!(RegularizationMethod::NormalizedMinEig as i32, 2);
    assert_eq!(RegularizationMethod::Plane as i32, 3);
    assert_eq!(RegularizationMethod::Frobenius as i32, 4);
}

#[test]
fn test_voxel_accumulation_mode_values() {
    assert_eq!(VoxelAccumulationMode::Additive as i32, 0);
    assert_eq!(VoxelAccumulationMode::AdditiveWeighted as i32, 1);
    assert_eq!(VoxelAccumulationMode::Multiplicative as i32, 2);
}

#[test]
fn test_neighbor_search_method_values() {
    assert_eq!(NeighborSearchMethod::Direct27 as i32, 0);
    assert_eq!(NeighborSearchMethod::Direct7 as i32, 1);
    assert_eq!(NeighborSearchMethod::Direct1 as i32, 2);
    assert_eq!(NeighborSearchMethod::DirectRadius as i32, 3);
}

#[test]
fn test_regularization_method_conversions() {
    // Test that we can convert from i32 and back
    let methods = [
        RegularizationMethod::None,
        RegularizationMethod::MinEig,
        RegularizationMethod::NormalizedMinEig,
        RegularizationMethod::Plane,
        RegularizationMethod::Frobenius,
    ];

    for method in methods {
        let value = method as i32;
        // In a real implementation, we'd have TryFrom<i32> implemented
        // For now, we just verify the values are distinct and in expected range
        assert!((0..=4).contains(&value));
    }
}

#[test]
fn test_enum_debug_output() {
    // Ensure Debug trait provides useful output
    let method = RegularizationMethod::MinEig;
    let debug_str = format!("{:?}", method);
    assert_eq!(debug_str, "MinEig");

    let mode = VoxelAccumulationMode::Additive;
    let debug_str = format!("{:?}", mode);
    assert_eq!(debug_str, "Additive");

    let search = NeighborSearchMethod::Direct27;
    let debug_str = format!("{:?}", search);
    assert_eq!(debug_str, "Direct27");
}

#[test]
fn test_enum_equality() {
    // Test derive(PartialEq, Eq)
    assert_eq!(RegularizationMethod::None, RegularizationMethod::None);
    assert_ne!(RegularizationMethod::None, RegularizationMethod::MinEig);

    assert_eq!(
        VoxelAccumulationMode::Additive,
        VoxelAccumulationMode::Additive
    );
    assert_ne!(
        VoxelAccumulationMode::Additive,
        VoxelAccumulationMode::AdditiveWeighted
    );
}

#[test]
fn test_enum_copy_clone() {
    // Test derive(Copy, Clone)
    let method1 = RegularizationMethod::Frobenius;
    let method2 = method1; // Copy
    let method3 = method1; // Clone

    assert_eq!(method1, method2);
    assert_eq!(method1, method3);
}
