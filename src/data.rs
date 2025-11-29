use crate::runtime::Env;
use obwio::*;

/// Buffer is a struct that holds, well, a buffer.
/// It is returned by Buffer::new().
///
/// # Examples
///
/// ```rust
/// use obrah::data::Buffer;
/// use obrah::runtime::Env;
///
/// let mut env = Env::new(0, 0);
/// let mut data = vec![1.0f32; 10];
/// let buf = Buffer::new(&mut env, &mut data);
/// ```

pub struct Buffer<T>
where
    T: Clone + 'static,
{
    pub buffer: cl_mem,
    pub data: Vec<T>,
}

impl<T> Buffer<T>
where
    T: Clone + 'static,
{
    pub fn to(&mut self, env: &mut Env) {
        let mut data = self.data.clone();
        to_gpu(env, &mut data, self);
    }
    pub fn from(&mut self, data: &mut [T], env: &mut Env)
    where
        T: Copy + 'static,
    {
        from_gpu(env, data, self);
    }
    pub fn new(env: &mut Env, data: &[T]) -> Buffer<T>
    where
        T: Clone + 'static,
    {
        buffer_write(env, data)
    }
}

impl<T> Drop for Buffer<T>
where
    T: Clone + 'static,
{
    fn drop(&mut self) {
        cleanvar(self);
    }
}

/// Create a buffer in order to be sent to the GPU. This function is used implicitly by Buffer::new().
/// You don't have to call it.
fn buffer_write<T>(env: &mut Env, data: &[T]) -> Buffer<T>
where
    T: Clone + 'static,
{
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        let buf = clCreateBuffer(
            env.context,
            CL_MEM_READ_WRITE.into(),
            size,
            std::ptr::null_mut(),
            &mut env.err,
        );
        if env.err != 0 {
            panic!("OpenCL error: {}", env.err);
        }

        Buffer {
            buffer: buf,
            data: data.to_vec(),
        }
    }
}

/// Send the buffer, and the data supposed to be in the buffer, to the GPU.
/// Used by Buffer.to().
fn to_gpu<T>(env: &mut Env, data: &mut [T], buffer: &mut Buffer<T>)
where
    T: Clone + 'static,
{
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueWriteBuffer(
            env.queue,
            buffer.buffer,
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
/// Used by Buffer.from().
fn from_gpu<T>(env: &mut Env, data: &mut [T], buf: &mut Buffer<T>)
where
    T: Copy + 'static,
{
    unsafe {
        let size = data.len() * std::mem::size_of::<f32>();
        clEnqueueReadBuffer(
            env.queue,
            buf.buffer,
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

/// cleanvar() cleans the buffer. This function is used automatically by Drop for the Buffer struct.
///
/// # Examples
///
/// ```rust
/// use obrah::data::Buffer;
/// use obrah::runtime::Env;
/// {
///     let mut env = Env::new(0, 0);
///     let mut data = vec![1.0f32; 10];
///     let buf = Buffer::new(&mut env, &mut data);
/// } // <- automatically dropped here
/// ```
fn cleanvar<T>(buf: &mut Buffer<T>)
where
    T: Clone + 'static,
{
    unsafe {
        clReleaseMemObject(buf.buffer);
    }
}
