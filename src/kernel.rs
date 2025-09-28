use obwio::*;
use crate::runtime::Env;
use std::ffi::CString;


/// setarg() sets an argument. To make sure it works properly, initialize your variables in the order they are
/// used in the kernel.
pub fn setarg(env: &mut Env, arg_idx: usize) {
    unsafe {
        let size = std::mem::size_of::<cl_mem>();
        let buf_ptr: *const cl_mem = &env.buffers[arg_idx];
        clSetKernelArg(env.kernel, arg_idx as u32, size, buf_ptr as *const _);
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