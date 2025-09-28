use obwio::*;
use crate::runtime::Env;


/// Create a buffer in order to be sent to the GPU.
pub fn buffer_write(env: &mut Env, data: &[f32]) -> usize {
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        let buf = clCreateBuffer(env.context, CL_MEM_READ_WRITE.into(), size, std::ptr::null_mut(), &mut env.err);
        env.buffers.push(buf);
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
        env.buffers.len() - 1
    }
}

/// Send the buffer, and the data supposed to be in the buffer, to the GPU.
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
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}


/// Get data from the GPU, from a specific buffer.
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
        
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }
    }
}
