use std::{env, path::PathBuf};
#[cfg(feature = "cuda")]
use std::{path::Path, process::Command};

/// Find CUDA installation directory
#[cfg(feature = "cuda")]
fn find_cuda_root() -> Option<String> {
    // First, check environment variables
    if let Ok(cuda_root) = env::var("CUDA_ROOT") {
        if Path::new(&cuda_root).exists() {
            return Some(cuda_root);
        }
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        if Path::new(&cuda_home).exists() {
            return Some(cuda_home);
        }
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        if Path::new(&cuda_path).exists() {
            return Some(cuda_path);
        }
    }

    // Check if user specified a preferred CUDA version
    let preferred_version = env::var("CUDA_VERSION").ok();

    // Check common installation locations - CUDA 12.x only
    let common_paths = vec![
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.7",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.5",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.3",
        "/usr/local/cuda-12.2",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.0",
        "/opt/cuda",
        "/usr/cuda",
    ];

    // If preferred version is specified, check it first
    if let Some(version) = preferred_version {
        let preferred_path = format!("/usr/local/cuda-{}", version);
        if Path::new(&preferred_path).exists()
            && Path::new(&format!("{}/bin/nvcc", preferred_path)).exists()
        {
            println!("cargo:warning=Using preferred CUDA version: {}", version);
            return Some(preferred_path);
        }
    }

    // Find all available CUDA installations
    let mut available_cuda = Vec::new();
    for path in common_paths {
        if Path::new(path).exists() && Path::new(&format!("{}/bin/nvcc", path)).exists() {
            // Try to get version
            if let Ok(output) = Command::new(format!("{}/bin/nvcc", path))
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    let version_output = String::from_utf8_lossy(&output.stdout);
                    if let Some(version_line) =
                        version_output.lines().find(|l| l.contains("release"))
                    {
                        println!("cargo:warning=Found CUDA at {}: {}", path, version_line);
                    }
                }
            }
            available_cuda.push(path.to_string());
        }
    }

    // Return the first available CUDA installation
    if !available_cuda.is_empty() {
        return Some(available_cuda[0].clone());
    }

    // Try to find nvcc in PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Some(parent) = Path::new(&nvcc_path).parent() {
                if let Some(cuda_root) = parent.parent() {
                    return Some(cuda_root.to_string_lossy().to_string());
                }
            }
        }
    }

    None
}

/// Detect available CUDA architectures
#[cfg(feature = "cuda")]
fn detect_cuda_architectures() -> Option<String> {
    // First try environment variable
    if let Ok(archs) = env::var("CUDA_ARCHITECTURES") {
        return Some(archs);
    }

    // Try to detect from nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let caps = String::from_utf8_lossy(&output.stdout);
            let architectures: Vec<String> = caps
                .lines()
                .filter_map(|line| {
                    let cap = line.trim().replace(".", "");
                    if !cap.is_empty() {
                        Some(cap)
                    } else {
                        None
                    }
                })
                .collect();

            if !architectures.is_empty() {
                return Some(architectures.join(";"));
            }
        }
    }

    // Return None to use defaults
    None
}

fn main() {
    // Tell cargo to rerun this build script if the wrapper changes
    println!("cargo:rerun-if-changed=src/wrapper.cpp");
    println!("cargo:rerun-if-changed=include/wrapper.h");
    println!("cargo:rerun-if-changed=../fast_gicp");

    // Get output directory
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check for system dependencies
    let pcl = pkg_config::Config::new()
        .atleast_version("1.8")
        .probe("pcl_common-1.12")
        .expect("PCL library not found. Please install PCL development package.");

    // Probe for Eigen3
    let eigen = pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("eigen3")
        .expect("Eigen3 library not found. Please install Eigen3 development package.");

    println!("cargo:rustc-link-lib=pcl_common");
    println!("cargo:rustc-link-lib=pcl_io");
    println!("cargo:rustc-link-lib=pcl_search");
    println!("cargo:rustc-link-lib=pcl_registration");
    println!("cargo:rustc-link-lib=flann");
    println!("cargo:rustc-link-lib=flann_cpp");

    // Configure CMAKE
    let mut cmake_config = cmake::Config::new("../fast_gicp");

    // Basic CMake configuration
    cmake_config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_apps", "OFF")
        .define("BUILD_test", "OFF")
        .define("BUILD_PYTHON_BINDINGS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");

    // Handle CUDA feature
    #[cfg(feature = "cuda")]
    {
        cmake_config.define("BUILD_VGICP_CUDA", "ON");

        // Probe for CUDA installation to set up library paths
        let cuda_root = find_cuda_root();
        if let Some(cuda_path) = &cuda_root {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        }

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=curand");
        println!("cargo:rustc-link-lib=cusparse");
    }

    #[cfg(not(feature = "cuda"))]
    cmake_config.define("BUILD_VGICP_CUDA", "OFF");

    // Add PCL include paths
    for include_path in &pcl.include_paths {
        cmake_config.cflag(format!("-I{}", include_path.display()));
        cmake_config.cxxflag(format!("-I{}", include_path.display()));
    }

    // Add Eigen include paths
    let mut eigen_include_paths = Vec::new();
    for include_path in &eigen.include_paths {
        cmake_config.cflag(format!("-I{}", include_path.display()));
        cmake_config.cxxflag(format!("-I{}", include_path.display()));
        eigen_include_paths.push(include_path.display().to_string());
    }

    // For CUDA compilation, set include directories properly
    #[cfg(feature = "cuda")]
    {
        // Probe for CUDA installation
        let cuda_root = find_cuda_root();

        // Set Eigen path for CMake to find automatically
        if let Some(eigen_path) = eigen_include_paths.first() {
            cmake_config.define("EIGEN3_INCLUDE_DIR", eigen_path);
            cmake_config.define("CMAKE_PREFIX_PATH", eigen_path);
        }
        if let Some(cuda_path) = &cuda_root {
            println!("cargo:warning=Found CUDA at: {}", cuda_path);

            // Configure CUDA toolkit paths
            cmake_config.define("CMAKE_CUDA_COMPILER", format!("{}/bin/nvcc", cuda_path));
            cmake_config.define("CUDA_TOOLKIT_ROOT_DIR", cuda_path);
            cmake_config.define(
                "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES",
                format!("{}/include", cuda_path),
            );
            cmake_config.define("CUDAToolkit_ROOT", cuda_path);

            // Set environment variables for CUDA
            cmake_config.env("CUDACXX", format!("{}/bin/nvcc", cuda_path));
            cmake_config.env("CUDA_ROOT", cuda_path);
            cmake_config.env(
                "PATH",
                format!("{}/bin:{}", cuda_path, env::var("PATH").unwrap_or_default()),
            );
        } else {
            panic!("CUDA installation not found. Please install CUDA toolkit or set CUDA_ROOT environment variable.");
        }

        // Configure CUDA compilation flags for CUDA 12.x
        let cuda_flags = [
            "-diag-suppress 20012",              // Suppress Eigen attribute warnings
            "--expt-relaxed-constexpr",          // Required for Eigen/Thrust compatibility
            "--extended-lambda",                 // Required for CUDA lambdas
            "-std=c++17",                        // Use C++17 for CUDA 12.x
            "--use_fast_math",                   // Optimize math operations
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK", // Ignore CUB version mismatches
            "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA", // Force CUDA backend
            "-DCUDA_API_PER_THREAD_DEFAULT_STREAM", // Use per-thread default stream
            "-DFAST_GICP_CUDA_12_MODERNIZATION", // Enable our modernization code
            "--disable-warnings",                // Reduce noise during migration
        ];
        cmake_config.define("CMAKE_CUDA_FLAGS", cuda_flags.join(" "));

        // Set CUDA standard for CUDA 12.x compatibility
        cmake_config.define("CMAKE_CUDA_STANDARD", "17");
        cmake_config.define("CMAKE_CUDA_STANDARD_REQUIRED", "ON");
        cmake_config.define("CMAKE_CXX_STANDARD", "17");
        cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");

        // Detect GPU architectures or use reasonable defaults for modern GPUs
        let cuda_archs = detect_cuda_architectures().unwrap_or_else(|| {
            // Support only recent GPU architectures for faster builds (Turing, Ampere, Ada Lovelace, Hopper)
            "75;80;86;87;89;90".to_string()
        });
        cmake_config.define("CMAKE_CUDA_ARCHITECTURES", cuda_archs);
    }

    // Build fast_gicp (note: no install target, just build)
    let fast_gicp_build = cmake_config.build_target("fast_gicp").build();

    // Tell cargo where to find the compiled library
    println!(
        "cargo:rustc-link-search=native={}/build",
        fast_gicp_build.display()
    );
    println!("cargo:rustc-link-lib=fast_gicp");

    // Build our wrapper using cxx-build
    let mut cxx_build = cxx_build::bridge("src/lib.rs");
    cxx_build
        .file("src/wrapper.cpp")
        .include("include")
        .include(format!("{}/include", fast_gicp_build.display()))
        .include("../fast_gicp/include")
        .includes(&eigen.include_paths)
        .includes(&pcl.include_paths)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-DFAST_GICP_CUDA_12_MODERNIZATION");

    // Add CUDA flag and include paths to CXX build
    #[cfg(feature = "cuda")]
    {
        cxx_build.flag_if_supported("-DBUILD_VGICP_CUDA");

        // Add CUDA include paths
        if let Some(cuda_path) = find_cuda_root() {
            cxx_build.include(format!("{}/include", cuda_path));
        }
    }

    #[cfg(not(feature = "cuda"))]
    cxx_build.flag_if_supported("-UBUILD_VGICP_CUDA");

    cxx_build.compile("fast_gicp_wrapper");

    // Generate bindings for rust-analyzer and IDE support
    println!("cargo:rustc-link-lib=fast_gicp_wrapper");
}
