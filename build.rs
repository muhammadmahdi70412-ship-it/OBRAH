fn main() {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
    } else {
        // Try to read the environment variable OPENCL_SDK.
        if let Ok(path) = std::env::var("OPENCL_SDK") {
            println!("cargo:rustc-link-search=native={}/lib", path);
            println!("cargo:rustc-link-lib=dylib=OpenCL");
        } else {
            println!("cargo:error=Environment variable OPENCL_SDK not set. Set it to your OpenCL SDK path.");
        }
    }
}