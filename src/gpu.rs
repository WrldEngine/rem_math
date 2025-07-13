// NOTE: Work in progress, will be refactored

extern crate ocl;
use ocl::{Buffer, MemFlags, ProQue};

const KERNEL_SRC: &'static str = include_str!("kernel.cl");

pub fn sum_two_ints32(arr_1: &[i32], arr_2: &[i32], result_vec: &mut Vec<i64>) {
    let pro_que = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(arr_1.len())
        .build()
        .unwrap();

    let buffer_1 = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(arr_1.len())
        .copy_host_slice(&arr_1)
        .build()
        .unwrap();

    let buffer_2 = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(arr_1.len())
        .copy_host_slice(&arr_2)
        .build()
        .unwrap();

    let result = pro_que.create_buffer::<i64>().unwrap();

    let kernel = pro_que
        .kernel_builder("add_i")
        .arg(&buffer_1)
        .arg(&buffer_2)
        .arg(&result)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    result.read(result_vec).enq().unwrap();
}

pub fn dot_float(arr_1: &[f32], arr_2: &[f32], result_vec: &mut Vec<f32>) {
    let pro_que = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(arr_1.len())
        .build()
        .unwrap();

    let buffer_1 = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(arr_1.len())
        .copy_host_slice(&arr_1)
        .build()
        .unwrap();

    let buffer_2 = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(arr_1.len())
        .copy_host_slice(&arr_2)
        .build()
        .unwrap();

    let result = pro_que.create_buffer::<f32>().unwrap();

    let kernel = pro_que
        .kernel_builder("dot_f")
        .arg(&buffer_1)
        .arg(&buffer_2)
        .arg(&result)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    result.read(result_vec).enq().unwrap();
}
