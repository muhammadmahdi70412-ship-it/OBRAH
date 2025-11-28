use obrah::data::*;
use obrah::kernel::*;
use obrah::runtime::*;
use std::io::Write;

fn lerp(start: f32, end: f32, total: i32) -> Vec<f32> {
    let step = (end - start) / total as f32;
    let mut last = Vec::with_capacity((total + 1) as usize);
    for i in 0..=total {
        last.push(start + step * i as f32);
    }
    last
}

fn lerp_sphere(
    start_x: f32,
    end_x: f32,
    start_y: f32,
    end_y: f32,
    start_z: f32,
    end_z: f32,
    start_rad: f32,
    end_rad: f32,
    frames: i32,
) -> Vec<[f32; 4]> {
    let mut end = Vec::with_capacity(frames as usize);
    let all_x = lerp(start_x, end_x, frames);
    let all_y = lerp(start_y, end_y, frames);
    let all_z = lerp(start_z, end_z, frames);
    let all_rad = lerp(start_rad, end_rad, frames);
    for i in 0..frames {
        end.push([
            all_x[i as usize],
            all_y[i as usize],
            all_z[i as usize],
            all_rad[i as usize],
        ]);
    }
    end
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    let mut env = Env::new(0, 0)?; // fix this with the right device - run example get_gpus to see all devices and platforms.
    env.use_kernel("examples/raytrace_kernel.cl")?
        .program()?
        .make_kernel("raytrace")?;

    let mut data = vec![0.0f32; WIDTH * HEIGHT * 4]; // we are going to make an empty buffer 1920 * 1080, and multiply by 4 for all 4 channels.
    let mut data_buf = Buffer::new(&mut env, &data);

    let light = [960.0f32, 540.0, 700.0];

    setarg(&env, &data_buf, 0); // our first argument is the empty buffer.
    setarg_scalar(&env, &WIDTH, 1); // the second argument is the width.
    setarg_scalar(&env, &HEIGHT, 2); // the third argument is the height.
    setarg_scalar(&env, &light, 4);

    let mut ffmpeg = std::process::Command::new("ffmpeg")
        .args([
            "-y", // overwrite output
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-r",
            "30", // framerate
            "-i",
            "-", // read from stdin
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "output.mp4",
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()?;

    let ffmpeg_stdin = ffmpeg.stdin.as_mut().unwrap();

    // now we are going to make an animation render thing
    let mut buf_frames: Vec<u8> = Vec::with_capacity(WIDTH * HEIGHT * 3 * 180); // poor ram
    let full_sphere = lerp_sphere(1920.0, 800.0, 0.0, 540.0, 0.0, 300.0, 256.0, 256.0, 180);
    let time = std::time::Instant::now();
    for i in 1..=180 {
        let sphere = full_sphere[(i - 1) as usize];
        setarg_scalar(&env, &sphere, 3);

        run_kernel(&mut env, WIDTH, HEIGHT); // we are going to run the kernel at 1080p

        data_buf.from(&mut data, &mut env); // now, we are going to retrieve the data from the buffer.
        for chunk in data.chunks_exact(4) {
            // we will get just the red, green, and blue values - no alpha in ppm.
            let r = (chunk[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (chunk[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (chunk[2].clamp(0.0, 1.0) * 255.0) as u8;
            buf_frames.extend_from_slice(&[r, g, b]);
        }

        println!("Saved frame {i}.");
    }

    ffmpeg_stdin.write_all(&buf_frames)?;
    ffmpeg_stdin.flush()?;
    drop(ffmpeg.stdin.take());

    ffmpeg.wait()?;

    println!("{:?} seconds to render video.", time.elapsed());

    println!("Output saved.");
    Ok(())
}
