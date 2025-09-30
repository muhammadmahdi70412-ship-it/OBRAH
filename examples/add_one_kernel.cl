__kernel void add_one(float a, __global float* b) {
    b[0] = a + 1.0f;
}
