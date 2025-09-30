use obrah::runtime::Env;
use obrah::data::Buffer;
use obrah::kernel::{make_kernel, setarg, run_kernel};

fn main() {
    // Setup
    let mut env = Env::new();
    env.use_kernel("examples/vecadd_kernel.cl");
    env.program();
    make_kernel(&mut env, "vec_add");

    let mut a = vec![7.0f32, 8.0, 2.0, 6.0];
    let mut b = vec![134.0f32, 134.11, 34.8, 112.9];
    let mut result = vec![0.0f32; a.len()];

    // Buffers
    let mut buf_a = Buffer::new(&mut env, &mut a);
    let mut buf_b = Buffer::new(&mut env, &mut b);
    let mut buf_result = Buffer::new(&mut env, &mut result);

    // Send data to GPU
    buf_a.to(&a, &mut env);
    buf_b.to(&b, &mut env);

    // Set kernel arguments
    setarg(&env, &mut buf_a, 0);
    setarg(&env, &mut buf_b, 1);
    setarg(&env, &mut buf_result, 2);

    // Run kernel
    run_kernel(&mut env, a.len());

    // Read result
    buf_result.from(&mut result, &mut env);

    println!("Result: {:?}", result);
}
