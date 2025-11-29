use obrah::data::Buffer;
use obrah::kernel::*;
use obrah::runtime::Env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup - Initialise kernel & queue.
    let mut env = Env::new(0, 0)?;
    env.use_kernel("examples/vecadd_kernel.cl")?
        .program()?
        .make_kernel("vec_add")?;

    // Create data.
    let a = vec![7.0f32, 8.0, 2.0, 6.0];
    let b = vec![134.0f32, 134.11, 34.8, 112.9];
    let mut result = vec![0.0f32; a.len()];

    // Buffers.
    let mut buf_a = Buffer::new(&mut env, &a);
    let mut buf_b = Buffer::new(&mut env, &b);
    let mut buf_result = Buffer::new(&mut env, &result);

    // Send data to GPU.
    buf_a.to(&mut env);
    buf_b.to(&mut env);

    // Set kernel arguments.
    setarg(&env, &buf_a, 0)?;
    setarg(&env, &buf_b, 1)?;
    setarg(&env, &buf_result, 2)?;

    // Run kernel.
    run_kernel(&mut env, a.len(), 1);

    // Read result.
    buf_result.from(&mut result, &mut env);

    // Output result.
    println!("Result: {:#?}", result);
    Ok(())
}
