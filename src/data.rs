use obwio::*;
use crate::runtime::Env;

pub fn buffer_write(env: &mut Env, data: &[f32]) -> usize {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        let buf = clCreateBuffer(env.context, CL_MEM_READ_WRITE.into(), size, std::ptr::null_mut(), &mut env.err);
        env.buffers.push(buf);
        env.buffers.len() - 1
    }
}

pub fn to_gpu(env: &mut Env, data: &[f32], idx: usize) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueWriteBuffer(
            env.queue,
            env.buffers[idx],
            CL_TRUE,
            0,
            size,
            data.as_ptr() as *const _,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
}

pub fn from_gpu(env: &mut Env, data: &mut [f32], idx: usize) {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueReadBuffer(
            env.queue,
            env.buffers[idx],
            CL_TRUE,
            0,
            size,
            data.as_mut_ptr() as *mut _,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
}
