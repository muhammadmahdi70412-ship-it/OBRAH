use obwio::*;
use crate::runtime::Env;
use std::ffi::CString;
use crate::data::Buffer;
use std::ffi::c_void;


/// setarg() sets an argument. For scalar values, use setarg_scalar().
/// The third parameter, arg, is the 0-based index of the kernel argument. 
/// 
/// # Examples
/// 
/// ```rust
/// use obrah::kernel;
/// use obrah::runtime;
/// use obrah::data;
/// use obrah::runtime::Env;
/// 
/// fn main() {
///     let mut env = Env::new();
///     env.use_kernel("examples/vecadd_kernel.cl");
///     env.program();
///     kernel::make_kernel(&mut env, "vec_add");
///     //kernel::make_kernel(&mut env, "kernel_name");
/// 
///     let mut a = vec![7.0f32, 8.0, 2.0, 6.0];
///     let mut b = vec![134.0f32, 134.11, 34.8, 112.9];
///     let mut result = vec![0.0f32; a.len()];
/// 
///     // Buffers
///     let mut buf_a = data::Buffer::new(&mut env, &mut a);
///     let mut buf_b = data::Buffer::new(&mut env, &mut b);
///     let mut buf_result = data::Buffer::new(&mut env, &mut result);
/// 
///      // Send data to GPU
///     buf_a.to(&a, &mut env);
///     buf_b.to(&b, &mut env);
/// 
///     // Set kernel arguments
///     kernel::setarg(&env, &mut buf_a, 0);
///     kernel::setarg(&env, &mut buf_b, 1);
///     kernel::setarg(&env, &mut buf_result, 2);
/// }
/// ```
/// 
pub fn setarg(env: &Env, buffer: &Buffer, arg: usize) {
    unsafe {
        let size = std::mem::size_of::<cl_mem>();
        let buf_ptr: *const std::ffi::c_void = &buffer.buffer as *const _ as *const _;
        clSetKernelArg(env.kernel, arg as u32, size, buf_ptr as *const _);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}


use std::ffi::CStr;
use std::ptr;

pub fn firstdevice() {
    unsafe {
        // Get platform
        let mut num_platforms = 0;
        clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms);
        let mut platforms = vec![std::ptr::null_mut(); num_platforms as usize];
        clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), ptr::null_mut());

        // Get devices
        let mut num_devices = 0;
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL.into(), 0, ptr::null_mut(), &mut num_devices);
        let mut devices = vec![std::ptr::null_mut(); num_devices as usize];
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL.into(), num_devices, devices.as_mut_ptr(), ptr::null_mut());

        // Get device name
        let mut size = 0;
        clGetDeviceInfo(devices[1], CL_DEVICE_NAME, 0, ptr::null_mut(), &mut size);
        let mut name_buf = vec![0u8; size];
        clGetDeviceInfo(devices[1], CL_DEVICE_NAME, size, name_buf.as_mut_ptr() as *mut _, ptr::null_mut());

        let name = CStr::from_bytes_with_nul(&name_buf).unwrap();
        println!("First OpenCL device: {}", name.to_str().unwrap());
    }
}

/// Set a scalar argument.
/// Scalar arguments are single-data types, such as
/// floats and integers.
/// They cannot be read from.
pub fn setarg_scalar<T>(env: &Env, val: &T, arg: usize) {
    unsafe {
        let size = std::mem::size_of::<T>();
        clSetKernelArg(env.kernel, arg as u32, size, val as *const T as *const c_void);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}

/// Run the kernel! Simply input the number of threads.
pub fn run_kernel(env: &mut Env, threads: usize) {
    unsafe {
        let global_work_size: [usize; 1] = [threads];
        clEnqueueNDRangeKernel(
            env.queue,
            env.kernel,
            1,
            std::ptr::null(),
            global_work_size.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
}


/// Make the kernel.
pub fn make_kernel(env: &mut Env, name: &str) {
    unsafe {
        let cname = CString::new(name).unwrap();
        env.kernel = clCreateKernel(env.program, cname.as_ptr(), &mut env.err);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}