use obwio::*;
use crate::runtime::Env;

pub fn setarg(env: &mut Env, idx: usize) {
    unsafe {
        let size = std::mem::size_of::<cl_mem>();
        let buf_ptr: *const cl_mem = &env.buffers[idx];
        clSetKernelArg(env.kernel, 0, size, buf_ptr as *const _);
    }
}

pub fn run_kernel(env: &mut Env, threads: usize) {
    unsafe {
        clEnqueueNDRangeKernel(env.queue, env.kernel, 1, std::ptr::null_mut(), &threads, std::ptr::null_mut(), 0, std::ptr::null_mut(), std::ptr::null_mut());
    }
}