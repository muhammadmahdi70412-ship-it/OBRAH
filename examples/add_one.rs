use obrah::runtime;
use obrah::data;
use obrah::kernel;
fn main() {
    let mut input = String::new();

    println!("Enter number to add one to: ");
    std::io::stdin()
        .read_line(&mut input)
        .expect("Failed");

    let trimmed = input.trim();

    let floaty = match trimmed.parse::<f32>() {
        Ok(val) => val,
        Err(e) => panic!("Error: {}", e)
    };


    let mut env = runtime::Env::new(false);
    env.use_kernel("examples/add_one_kernel.cl");
    env.program();
    kernel::make_kernel(&mut env, "add_one");

    let mut b = vec![0.0f32];
    let mut bbuf = data::Buffer::new(&mut env, &mut b);
    bbuf.to(&b, &mut env);

    let a = floaty;

    kernel::setarg_scalar(&env, &a, 0);
    kernel::setarg(&env, &mut bbuf, 1);



    kernel::run_kernel(&mut env, 1);

    bbuf.from(&mut b, &mut env);

    println!("Result: {:?}", b[0]);

}