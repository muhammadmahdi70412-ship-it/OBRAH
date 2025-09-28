fn main() {
    println!("cargo:rustc-link-search=native=D:/ashha/Downloads/OpenCL-SDK-v2025.07.23-Win-x64/OpenCL-SDK-v2025.07.23-Win-x64/lib");
    println!("cargo:rustc-link-lib=dylib=OpenCL");
}
