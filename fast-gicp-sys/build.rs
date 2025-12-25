use std::path::PathBuf;

fn main() {
    // Validate exclusive features
    if cfg!(all(feature = "bindgen", feature = "docs-only")) {
        panic!("bindgen and docs-only features are mutually exclusive");
    }

    // Skip C++ compilation if docs-only is enabled
    if cfg!(feature = "docs-only") {
        println!(
            "cargo:warning=docs-only mode: skipping C++ compilation, using pre-generated bindings"
        );
        return;
    }

    // Normal build: compile C++ wrapper and link system libraries
    println!("cargo:warning=Compiling C++ wrapper code");

    // Check for system libraries first
    let pcl = check_system_libraries();

    // Check CUDA availability
    let cuda_enabled = if cfg!(feature = "cuda") {
        match probe_cuda() {
            Ok(_) => {
                println!("cargo:warning=CUDA support enabled");
                true
            }
            Err(e) => {
                println!("cargo:warning=CUDA setup failed: {e}");
                println!("cargo:warning=Building without CUDA support");
                false
            }
        }
    } else {
        false
    };

    compile_cpp_wrapper(&pcl, cuda_enabled);
    link_system_libraries(&pcl, cuda_enabled);

    // Regenerate stubs if bindgen feature is enabled
    if cfg!(feature = "bindgen") {
        println!("cargo:warning=Regenerating stubs from C++ headers");
        generate_stubs();
    }
}

/// Check for required system libraries
fn check_system_libraries() -> pkg_config::Library {
    // Check for PCL
    pkg_config::Config::new()
        .atleast_version("1.8")
        .probe("pcl_common-1.12")
        .expect("PCL library not found. Please install PCL development package.")
}

/// Compile the C++ wrapper code using vendored sources
fn compile_cpp_wrapper(pcl: &pkg_config::Library, cuda_enabled: bool) {
    let fast_gicp_dir = get_fast_gicp_dir();

    // Build our wrapper using cxx-build
    // Always use bridge.rs for C++ compilation to ensure headers are generated correctly
    let mut cxx_build = cxx_build::bridge("src/bridge.rs");
    cxx_build
        .file("src/wrapper.cpp")
        .include("include")
        .include(fast_gicp_dir.join("include"))
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-fopenmp")
        .flag_if_supported("-DFAST_GICP_CUDA_12_MODERNIZATION");

    // Add CUDA flag and include paths only if CUDA is actually available
    if !cuda_enabled {
        cxx_build.flag_if_supported("-UBUILD_VGICP_CUDA");
    } else {
        cxx_build.flag_if_supported("-DBUILD_VGICP_CUDA");

        // Add CUDA include paths
        #[cfg(feature = "cuda")]
        {
            let Ok(cuda_info) = probe_cuda() else {
                // This shouldn't happen since cuda_enabled is true, but handle gracefully
                return;
            };

            let cuda_include = format!("{}/include", cuda_info.root);
            cxx_build.include(&cuda_include);
            println!("cargo:warning=Added CUDA include: {cuda_include}");
        }
    }

    // Add PCL include paths
    for include_path in &pcl.include_paths {
        cxx_build.include(include_path);
    }
    println!("cargo:warning=Using system PCL");

    // Try to find and use system Eigen if available, otherwise use vendored
    if let Ok(eigen) = pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("eigen3")
    {
        for include_path in &eigen.include_paths {
            cxx_build.include(include_path);
        }
        println!("cargo:warning=Using system Eigen3");
    } else {
        // Try vendored Eigen
        let eigen_dir = fast_gicp_dir.join("thirdparty");
        if eigen_dir.exists() {
            cxx_build.include(eigen_dir);
            println!("cargo:warning=Using vendored Eigen3 (if available)");
        } else {
            println!("cargo:warning=No Eigen3 found - compilation may fail");
        }
    }

    cxx_build.compile("fast_gicp_wrapper");

    println!("cargo:rustc-link-lib=fast_gicp_wrapper");
}

/// Generate stubs from C++ headers (requires system dependencies)
fn generate_stubs() {
    use std::fs;
    use syn::parse_file;

    println!("cargo:warning=Parsing cxx::bridge definition from lib.rs");

    // Parse the cxx::bridge definition directly from lib.rs
    let lib_content = fs::read_to_string("src/lib.rs").expect("Failed to read src/lib.rs");
    let syntax = parse_file(&lib_content).expect("Failed to parse src/lib.rs");

    // Extract the cxx::bridge module
    let bridge_mod =
        find_cxx_bridge_module(&syntax).expect("Failed to find cxx::bridge module in lib.rs");

    println!("cargo:warning=Generating dual stub variants");

    // Create generated directory if it doesn't exist
    fs::create_dir_all("src/generated").expect("Failed to create generated directory");

    // Generate stub version without CUDA (docs-only)
    let stub = generate_stub_bindings(&bridge_mod, false);
    fs::write("src/generated/stub.rs", format_with_rustfmt(stub)).expect("Failed to write stub.rs");

    // Generate stub version with CUDA (docs-only + cuda)
    let stub_cuda = generate_stub_bindings(&bridge_mod, true);
    fs::write("src/generated/stub_cuda.rs", format_with_rustfmt(stub_cuda))
        .expect("Failed to write stub_cuda.rs");

    println!("cargo:warning=Generated stubs have been updated in src/generated/");
    println!("cargo:warning=Please verify with: cargo expand --lib > expanded.rs");
}

/// Find the cxx::bridge module in the parsed file
fn find_cxx_bridge_module(file: &syn::File) -> Option<syn::ItemMod> {
    use syn::Item;
    for item in &file.items {
        if let Item::Mod(item_mod) = item {
            // Look for the ffi module with cxx::bridge attribute
            if item_mod.ident == "ffi" {
                // It should have cxx::bridge attribute
                let has_cxx_bridge = item_mod.attrs.iter().any(|attr| {
                    if let syn::Meta::Path(path) = &attr.meta {
                        if let Some(seg) = path.segments.first() {
                            return seg.ident == "cxx";
                        }
                    }
                    false
                });

                if has_cxx_bridge {
                    return Some(item_mod.clone());
                }
            }
        }
    }
    None
}

/// Generate stub bindings with unreachable!() implementations
fn generate_stub_bindings(bridge_mod: &syn::ItemMod, include_cuda: bool) -> String {
    use quote::quote;
    use syn::{ForeignItem, ForeignItemType, Item};

    let mut stub_items = vec![];

    // Add imports - critically, we use cxx::UniquePtr
    stub_items.push(quote! {
        use std::pin::Pin;
        use cxx::UniquePtr;
    });

    // Process the module content
    if let Some((_, items)) = &bridge_mod.content {
        for item in items {
            match item {
                // Handle struct definitions (Transform4f, Point3f, Point4f)
                Item::Struct(s) => {
                    stub_items.push(quote! { #s });
                }

                // Handle the unsafe extern "C++" block
                Item::ForeignMod(foreign_mod) => {
                    for foreign_item in &foreign_mod.items {
                        match foreign_item {
                            // Opaque types: type Foo; → pub struct Foo { _private: [u8; 0] }
                            ForeignItem::Type(ForeignItemType { ident, .. }) => {
                                // Check if this is a CUDA-specific type
                                let is_cuda_type = ident.to_string().contains("Cuda");

                                // Skip CUDA types if not including CUDA
                                if is_cuda_type && !include_cuda {
                                    continue;
                                }

                                stub_items.push(quote! {
                                    #[repr(C)]
                                    pub struct #ident {
                                        _private: ::cxx::private::Opaque,
                                    }

                                    // Implement ExternType trait for cxx compatibility
                                    unsafe impl ::cxx::ExternType for #ident {
                                        type Id = ::cxx::type_id!(#ident);
                                        type Kind = ::cxx::kind::Opaque;
                                    }

                                    // Implement UniquePtrTarget trait - required for UniquePtr<T>
                                    unsafe impl ::cxx::private::UniquePtrTarget for #ident {
                                        fn __typename(f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                                            f.write_str(stringify!(#ident))
                                        }
                                        fn __null() -> ::core::mem::MaybeUninit<*mut ::core::ffi::c_void> {
                                            ::core::mem::MaybeUninit::new(::core::ptr::null_mut())
                                        }
                                        unsafe fn __raw(raw: *mut Self) -> ::core::mem::MaybeUninit<*mut ::core::ffi::c_void> {
                                            ::core::mem::MaybeUninit::new(raw as _)
                                        }
                                        unsafe fn __get(repr: ::core::mem::MaybeUninit<*mut ::core::ffi::c_void>) -> *const Self {
                                            repr.assume_init() as _
                                        }
                                        unsafe fn __release(repr: ::core::mem::MaybeUninit<*mut ::core::ffi::c_void>) -> *mut Self {
                                            repr.assume_init() as _
                                        }
                                        unsafe fn __drop(_repr: ::core::mem::MaybeUninit<*mut ::core::ffi::c_void>) {
                                            // For stub, do nothing
                                        }
                                    }
                                });
                            }

                            // Functions: generate stub with unreachable!()
                            ForeignItem::Fn(func) => {
                                // Check if function has cfg(feature = "cuda") attribute
                                let is_cuda_specific = func.attrs.iter().any(is_cuda_cfg_attribute);

                                // Skip CUDA items if not including CUDA
                                if is_cuda_specific && !include_cuda {
                                    continue;
                                }

                                let sig = &func.sig;
                                let attrs = &func.attrs;

                                // Always make functions public
                                stub_items.push(quote! {
                                    #(#attrs)*
                                    #[allow(unused_variables, dead_code)]
                                    pub #sig {
                                        unreachable!("docs-only stub")
                                    }
                                });
                            }

                            _ => {}
                        }
                    }
                }

                _ => {}
            }
        }
    }

    // Generate the ffi module
    quote! {
        /// FFI bindings stub for documentation generation.
        ///
        /// This module provides type definitions for documentation purposes when
        /// building on docs.rs where C++ dependencies are not available.
        pub mod ffi {
            #(#stub_items)*
        }
    }
    .to_string()
}

/// Check if an attribute is #[cfg(feature = "cuda")]
fn is_cuda_cfg_attribute(attr: &syn::Attribute) -> bool {
    // Check if this is a cfg attribute
    if !attr.path().is_ident("cfg") {
        return false;
    }

    // Parse the attribute arguments
    let mut is_cuda = false;
    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("feature") {
            let value = meta.value()?;
            let s: syn::LitStr = value.parse()?;
            if s.value() == "cuda" {
                is_cuda = true;
            }
        }
        Ok(())
    });

    is_cuda
}

/// Format code using rustfmt
fn format_with_rustfmt(code: String) -> String {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    let mut child = Command::new("rustfmt")
        .arg("--emit=stdout")
        .arg("--edition=2021")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn rustfmt");

    // Write code to rustfmt's stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(code.as_bytes())
            .expect("Failed to write to rustfmt");
        drop(stdin); // Close stdin to signal EOF
    }

    let output = child
        .wait_with_output()
        .expect("Failed to wait for rustfmt");

    if !output.status.success() {
        eprintln!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!("Using unformatted code");
        return code;
    }

    String::from_utf8(output.stdout).unwrap_or(code)
}

/// Link to system libraries (PCL, Eigen3, FLANN, etc.)
fn link_system_libraries(pcl: &pkg_config::Library, cuda_enabled: bool) {
    // Link PCL libraries
    println!("cargo:rustc-link-lib=pcl_common");
    println!("cargo:rustc-link-lib=pcl_io");
    println!("cargo:rustc-link-lib=pcl_search");
    println!("cargo:rustc-link-lib=pcl_registration");
    println!("cargo:rustc-link-lib=flann");
    println!("cargo:rustc-link-lib=flann_cpp");

    // Link LZ4 library (required for PCL IO compression)
    println!("cargo:rustc-link-lib=lz4");

    // Link OpenMP library (required for parallel algorithms in fast-gicp)
    // GNU OpenMP runtime on Linux
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=gomp");
    // Apple's libomp on macOS
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=omp");

    // Build fast_gicp library using CMake
    let fast_gicp_dir = get_fast_gicp_dir();
    let mut cmake_config = cmake::Config::new(&fast_gicp_dir);

    // Basic CMake configuration
    cmake_config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_apps", "OFF")
        .define("BUILD_test", "OFF")
        .define("BUILD_PYTHON_BINDINGS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");

    // Handle CUDA
    if cuda_enabled {
        cmake_config.define("BUILD_VGICP_CUDA", "ON");
        #[cfg(feature = "cuda")]
        {
            if let Err(e) = setup_cuda(&mut cmake_config) {
                println!("cargo:warning=CUDA CMake setup failed: {e}");
                // This shouldn't happen since we already probed CUDA, but handle gracefully
                cmake_config.define("BUILD_VGICP_CUDA", "OFF");
            }
        }
    } else {
        cmake_config.define("BUILD_VGICP_CUDA", "OFF");
    }

    // Add include paths
    for include_path in &pcl.include_paths {
        cmake_config.cflag(format!("-I{}", include_path.display()));
        cmake_config.cxxflag(format!("-I{}", include_path.display()));
    }

    // Build fast_gicp
    let fast_gicp_build = cmake_config.build_target("fast_gicp").build();

    // Link the built library
    println!(
        "cargo:rustc-link-search=native={}/build",
        fast_gicp_build.display()
    );

    // Static linking order matters: dependents before dependencies
    // fast_gicp depends on fast_vgicp_cuda (when CUDA is enabled)
    println!("cargo:rustc-link-lib=static=fast_gicp");
    if cuda_enabled {
        println!("cargo:rustc-link-lib=static=fast_vgicp_cuda");
    }
}

/// CUDA toolkit information
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaInfo {
    root: String,
    version: String,
}

/// Probe for CUDA installation using proper detection methods
#[cfg(feature = "cuda")]
fn probe_cuda() -> Result<CudaInfo, String> {
    use std::process::Command;

    // First try to find nvcc in PATH
    let Ok(output) = Command::new("nvcc").arg("--version").output() else {
        // nvcc not in PATH, try other methods
        return try_cuda_from_env_or_common_paths();
    };

    if !output.status.success() {
        return try_cuda_from_env_or_common_paths();
    }

    let version_output = String::from_utf8_lossy(&output.stdout);
    let Some(version) = extract_cuda_version(&version_output) else {
        return try_cuda_from_env_or_common_paths();
    };

    // Try to find CUDA root from nvcc location
    let Ok(which_output) = Command::new("which").arg("nvcc").output() else {
        return try_cuda_from_env_or_common_paths();
    };

    if !which_output.status.success() {
        return try_cuda_from_env_or_common_paths();
    }

    let nvcc_path_str = String::from_utf8_lossy(&which_output.stdout);
    let nvcc_path = nvcc_path_str.trim();

    // nvcc is typically in /path/to/cuda/bin/nvcc
    let Some(cuda_root) = std::path::Path::new(nvcc_path)
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.to_str())
    else {
        return try_cuda_from_env_or_common_paths();
    };

    Ok(CudaInfo {
        root: cuda_root.to_string(),
        version,
    })
}

/// Helper function to try CUDA detection from environment variables or common paths
#[cfg(feature = "cuda")]
fn try_cuda_from_env_or_common_paths() -> Result<CudaInfo, String> {
    use std::process::Command;

    // Try environment variables
    for env_var in ["CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"] {
        let Ok(cuda_root) = std::env::var(env_var) else {
            continue;
        };

        if !std::path::Path::new(&cuda_root).exists() {
            continue;
        }

        let nvcc_path = std::path::Path::new(&cuda_root).join("bin").join("nvcc");
        if !nvcc_path.exists() {
            continue;
        }

        // Verify nvcc works
        let Ok(output) = Command::new(&nvcc_path).arg("--version").output() else {
            continue;
        };

        if !output.status.success() {
            continue;
        }

        let version_output = String::from_utf8_lossy(&output.stdout);
        let Some(version) = extract_cuda_version(&version_output) else {
            continue;
        };

        return Ok(CudaInfo {
            root: cuda_root,
            version,
        });
    }

    // Try pkg-config (some distributions provide this)
    if let Ok(cuda) = pkg_config::Config::new().probe("cuda") {
        // Extract root from include paths (e.g., /usr/local/cuda/include -> /usr/local/cuda)
        for include_path in &cuda.include_paths {
            let Some(parent) = include_path.parent() else {
                continue;
            };

            let cuda_root = parent.to_string_lossy();
            let nvcc_path = parent.join("bin").join("nvcc");

            if !nvcc_path.exists() {
                continue;
            }

            let Ok(output) = Command::new(&nvcc_path).arg("--version").output() else {
                continue;
            };

            if !output.status.success() {
                continue;
            }

            let version_output = String::from_utf8_lossy(&output.stdout);
            let Some(version) = extract_cuda_version(&version_output) else {
                continue;
            };

            return Ok(CudaInfo {
                root: cuda_root.to_string(),
                version,
            });
        }
    }

    // Last resort: try common installation paths
    let common_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"];

    for &path in &common_paths {
        if !std::path::Path::new(path).exists() {
            continue;
        }

        let nvcc_path = std::path::Path::new(path).join("bin").join("nvcc");
        if !nvcc_path.exists() {
            continue;
        }

        let Ok(output) = Command::new(&nvcc_path).arg("--version").output() else {
            continue;
        };

        if !output.status.success() {
            continue;
        }

        let version_output = String::from_utf8_lossy(&output.stdout);
        let Some(version) = extract_cuda_version(&version_output) else {
            continue;
        };

        return Ok(CudaInfo {
            root: path.to_string(),
            version,
        });
    }

    Err(
        "CUDA installation not found. Please install CUDA toolkit or ensure nvcc is in PATH."
            .to_string(),
    )
}

#[cfg(not(feature = "cuda"))]
fn probe_cuda() -> Result<(), String> {
    Err("CUDA feature not enabled".to_string())
}

#[cfg(feature = "cuda")]
fn extract_cuda_version(nvcc_output: &str) -> Option<String> {
    // Parse version from nvcc output like "Cuda compilation tools, release 12.1, V12.1.105"
    for line in nvcc_output.lines() {
        if !line.contains("release") {
            continue;
        }

        let Some(start) = line.find("release ") else {
            continue;
        };

        let after_release = &line[start + 8..];
        let Some(end) = after_release.find(',') else {
            continue;
        };

        return Some(after_release[..end].trim().to_string());
    }
    None
}

/// Setup CUDA configuration
#[cfg(feature = "cuda")]
fn setup_cuda(cmake_config: &mut cmake::Config) -> Result<(), String> {
    let cuda_info = probe_cuda()?;
    let cuda_path = &cuda_info.root;

    println!("cargo:rustc-link-search=native={cuda_path}/lib64");
    println!("cargo:rustc-link-search=native={cuda_path}/lib");

    println!(
        "cargo:warning=Found CUDA {} at: {}",
        cuda_info.version, cuda_path
    );

    // Configure CUDA toolkit paths
    cmake_config.define("CMAKE_CUDA_COMPILER", format!("{cuda_path}/bin/nvcc"));
    cmake_config.define("CUDA_TOOLKIT_ROOT_DIR", cuda_path);
    cmake_config.define(
        "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES",
        format!("{cuda_path}/include"),
    );
    cmake_config.define("CUDAToolkit_ROOT", cuda_path);

    // Set environment variables
    cmake_config.env("CUDACXX", format!("{cuda_path}/bin/nvcc"));
    cmake_config.env("CUDA_ROOT", cuda_path);
    cmake_config.env(
        "PATH",
        format!(
            "{}/bin:{}",
            cuda_path,
            std::env::var("PATH").unwrap_or_default()
        ),
    );

    // Link CUDA libraries
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cusparse");

    // Configure CUDA compilation flags
    let cuda_flags = [
        "-diag-suppress 20012",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-std=c++17",
        "--use_fast_math",
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA",
        "-DCUDA_API_PER_THREAD_DEFAULT_STREAM",
        "-DFAST_GICP_CUDA_12_MODERNIZATION",
        "--disable-warnings",
    ];
    cmake_config.define("CMAKE_CUDA_FLAGS", cuda_flags.join(" "));

    // Set CUDA standard
    cmake_config.define("CMAKE_CUDA_STANDARD", "17");
    cmake_config.define("CMAKE_CUDA_STANDARD_REQUIRED", "ON");
    cmake_config.define("CMAKE_CXX_STANDARD", "17");
    cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");

    // Detect GPU architectures
    let cuda_archs = detect_cuda_architectures().unwrap_or_else(|| "75;80;86;87;89;90".to_string());
    cmake_config.define("CMAKE_CUDA_ARCHITECTURES", cuda_archs);

    Ok(())
}

/// Detect available GPU architectures
#[cfg(feature = "cuda")]
fn detect_cuda_architectures() -> Option<String> {
    use std::process::Command;

    // Try nvidia-smi
    let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    else {
        return None;
    };

    if !output.status.success() {
        return None;
    }

    let caps = String::from_utf8_lossy(&output.stdout);
    let architectures: Vec<String> = caps
        .lines()
        .filter_map(|line| {
            let cap = line.trim().replace('.', "");
            if cap.is_empty() {
                None
            } else {
                Some(cap)
            }
        })
        .collect();

    if architectures.is_empty() {
        None
    } else {
        Some(architectures.join(";"))
    }
}

/// Get the location of vendored fast_gicp sources
fn get_fast_gicp_dir() -> PathBuf {
    let vendor_dir = PathBuf::from("vendor/fast_gicp");

    if !vendor_dir.exists() || !vendor_dir.join(".vendor_ready").exists() {
        panic!(
            "Vendored fast_gicp sources not found!\n\
            \n\
            Please run the vendor preparation script first:\n\
            \n\
            From the project root:\n\
                ./scripts/prepare-vendor.sh\n\
            \n\
            Or from this directory:\n\
                ../scripts/prepare-vendor.sh\n\
            \n\
            This will copy the necessary C++ sources from the git submodule."
        );
    }

    println!(
        "cargo:warning=Using vendored fast_gicp sources from: {}",
        vendor_dir.display()
    );

    vendor_dir
}
