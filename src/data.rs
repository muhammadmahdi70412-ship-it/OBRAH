use obwio::*;
use crate::runtime::Env;

pub fn buffer_write(env: &mut Env, data: &[f32]) -> usize {
    unsafe {
        let size = std::mem::size_of_val(data);
        let buf = clCreateBuffer(env.context, CL_MEM_READ_WRITE.into(), size, std::ptr::null_mut(),&mut env.err,);
        env.buffers.push(buf);
        env.buffers.len() - 1
    }
}

pub fn to_gpu(env: &mut Env, data: &[f32], idx: usize) {
    unsafe {
        clEnqueueWriteBuffer(env.queue, env.buffers[idx], CL_TRUE, 0, std::mem::size_of_val(data), data.as_ptr() as *const _, 0, std::ptr::null_mut(), std::ptr::null_mut());
    }
}

pub fn from_gpu(env: &mut Env, data: &mut[f32], idx: usize) {
    unsafe {
        clEnqueueReadBuffer(env.queue, env.buffers[idx], CL_TRUE, 0, std::mem::size_of_val(data), data.as_mut_ptr() as *mut _, 0, std::ptr::null_mut(), std::ptr::null_mut());
    }
}