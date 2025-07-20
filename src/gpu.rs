// NOTE: Work in progress, will be refactored

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{
    Buffer, CL_MAP_WRITE, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
};
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::types::{
    cl_double, cl_event, cl_float, cl_int, cl_long, CL_BLOCKING, CL_NON_BLOCKING,
};

use std::ptr;

const KERNEL_SRC: &'static str = include_str!("kernel.cl");

pub struct GPUKernelsDispatcher {
    context: Context,
    program: Program,
    queue: CommandQueue,
}

impl GPUKernelsDispatcher {
    pub fn new() -> Self {
        let device_id: *mut std::ffi::c_void = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap()
            .first()
            .expect("no device found in platform");

        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("Context::from_device failed");

        let program = Program::create_and_build_from_source(&context, KERNEL_SRC, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed");

        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .expect("CommandQueue::create_default_with_properties failed");

        Self {
            context,
            program,
            queue,
        }
    }

    pub fn sum_two_ints32(&self, arr_1: &[i32], arr_2: &[i32], result_vec: &mut Vec<i64>) {
        let kernel = Kernel::create(&self.program, "add_i").expect("Kernel::create failed");

        let mut arr_1_buf = unsafe {
            Buffer::<cl_int>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                arr_1.len(),
                ptr::null_mut(),
            )
            .expect("allocation error")
        };
        let mut arr_2_buf = unsafe {
            Buffer::<cl_int>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                arr_2.len(),
                ptr::null_mut(),
            )
            .expect("allocation error")
        };
        let result_buf = unsafe {
            Buffer::<cl_long>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                result_vec.len(),
                ptr::null_mut(),
            )
            .expect("allocation error")
        };

        let _arr_1_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_1_buf, CL_NON_BLOCKING, 0, &arr_1, &[])
                .unwrap()
        };
        let _arr_2_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_2_buf, CL_NON_BLOCKING, 0, &arr_2, &[])
                .unwrap()
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&arr_1_buf)
                .set_arg(&arr_2_buf)
                .set_arg(&result_buf)
                .set_global_work_size(arr_1.len())
                .set_wait_event(&_arr_1_buf_write_event)
                .set_wait_event(&_arr_2_buf_write_event)
                .enqueue_nd_range(&self.queue)
                .unwrap()
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&result_buf, CL_NON_BLOCKING, 0, result_vec, &events)
                .unwrap()
        };

        read_event.wait().unwrap();
    }

    pub fn dot_floats32(&self, arr_1: &[f32], arr_2: &[f32]) -> f32 {
        let kernel = Kernel::create(&self.program, "dot_f").expect("Kernel::create failed");

        let mut arr_1_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                arr_1.len(),
                ptr::null_mut(),
            )
            .expect("opencl: allocation error")
        };

        let mut arr_2_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                arr_2.len(),
                ptr::null_mut(),
            )
            .expect("opencl: allocation error")
        };

        let local_size = 64;
        let group_count = (arr_1.len() + local_size - 1) / local_size;

        let partial_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                group_count,
                ptr::null_mut(),
            )
            .unwrap()
        };

        let _arr_1_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_1_buf, CL_NON_BLOCKING, 0, &arr_1, &[])
                .unwrap()
        };
        let _arr_2_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_2_buf, CL_NON_BLOCKING, 0, &arr_2, &[])
                .unwrap()
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&arr_1_buf)
                .set_arg(&arr_2_buf)
                .set_arg(&partial_buf)
                .set_global_work_size(arr_1.len())
                .set_local_work_size(local_size)
                .set_wait_event(&_arr_1_buf_write_event)
                .set_wait_event(&_arr_2_buf_write_event)
                .enqueue_nd_range(&self.queue)
                .unwrap()
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let mut partial_results = vec![0.0f32; group_count];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&partial_buf, CL_BLOCKING, 0, &mut partial_results, &[])
                .unwrap()
        };

        let result: f32 = partial_results.iter().sum();
        read_event.wait().unwrap();

        result
    }
}
