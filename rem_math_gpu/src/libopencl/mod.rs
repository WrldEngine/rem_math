use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR, CL_MEM_WRITE_ONLY};
use opencl3::program::{CL_STD_2_0, Program};
use opencl3::types::{CL_NON_BLOCKING, cl_event, cl_float, cl_int, cl_long};

use std::os::raw::c_void;
use std::ptr;

pub struct GPUKernelsDispatcher {
    context: Context,
    queue: CommandQueue,
}

impl GPUKernelsDispatcher {
    pub fn new() -> anyhow::Result<Self> {
        let device_id: *mut std::ffi::c_void = *get_all_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("no device found in platform");

        let device = Device::new(device_id);
        let context = Context::from_device(&device).expect("Context::from_device failed");
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .expect("CommandQueue::create_default_with_properties failed");

        Ok(Self { context, queue })
    }

    pub fn sum_two_ints32(
        &self,
        arr_1: &[i32],
        arr_2: &[i32],
        result_vec: &mut Vec<i64>,
    ) -> anyhow::Result<()> {
        const KERNEL_SRC: &'static str = include_str!("kernels/sum_ints.cl");

        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, CL_STD_2_0)
            .expect("failed to build kernel fromo source");
        let kernel = Kernel::create(&program, "sum_ints")?;

        let mut arr_1_buf = unsafe {
            Buffer::<cl_int>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_1.len(),
                arr_1.as_ptr() as *mut c_void,
            )?
        };

        let mut arr_2_buf = unsafe {
            Buffer::<cl_int>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_2.len(),
                arr_2.as_ptr() as *mut c_void,
            )?
        };

        let result_buf = unsafe {
            Buffer::<cl_long>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                result_vec.len(),
                ptr::null_mut(),
            )?
        };

        let _arr_1_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_1_buf, CL_NON_BLOCKING, 0, &arr_1, &[])?
        };
        let _arr_2_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_2_buf, CL_NON_BLOCKING, 0, &arr_2, &[])?
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&arr_1_buf)
                .set_arg(&arr_2_buf)
                .set_arg(&result_buf)
                .set_global_work_size(arr_1.len())
                .set_wait_event(&_arr_1_buf_write_event)
                .set_wait_event(&_arr_2_buf_write_event)
                .enqueue_nd_range(&self.queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&result_buf, CL_NON_BLOCKING, 0, result_vec, &events)?
        };

        read_event.wait()?;

        Ok(())
    }

    pub fn mul_mat(
        &self,
        m: usize,
        n: usize,
        k: usize,
        arr_1: &[f32],
        arr_2: &[f32],
        result_vec: &mut [f32],
    ) -> anyhow::Result<()> {
        const KERNEL_SRC: &'static str = include_str!("kernels/mul_mat_floats.cl");

        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, CL_STD_2_0)
            .expect("failed to build kernel fromo source");
        let kernel = Kernel::create(&program, "mul_mat_floats")?;

        let mut arr_1_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_1.len(),
                arr_1.as_ptr() as *mut c_void,
            )?
        };

        let mut arr_2_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_1.len(),
                arr_2.as_ptr() as *mut c_void,
            )?
        };

        let result_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                result_vec.len(),
                ptr::null_mut(),
            )?
        };

        let _arr_1_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_1_buf, CL_NON_BLOCKING, 0, &arr_1, &[])?
        };
        let _arr_2_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_2_buf, CL_NON_BLOCKING, 0, &arr_2, &[])?
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&m)
                .set_arg(&n)
                .set_arg(&k)
                .set_arg(&arr_1_buf)
                .set_arg(&arr_2_buf)
                .set_arg(&result_buf)
                .set_global_work_size(arr_1.len())
                .set_wait_event(&_arr_1_buf_write_event)
                .set_wait_event(&_arr_2_buf_write_event)
                .enqueue_nd_range(&self.queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&result_buf, CL_NON_BLOCKING, 0, result_vec, &events)?
        };

        read_event.wait()?;

        Ok(())
    }

    pub fn dot_floats32(&self, arr_1: &[f32], arr_2: &[f32]) -> anyhow::Result<f32> {
        const KERNEL_SRC: &'static str = include_str!("kernels/dot_floats.cl");

        let program = Program::create_and_build_from_source(&self.context, KERNEL_SRC, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed");

        let kernel = Kernel::create(&program, "dot_f")?;
        let arr_sz: cl_int = arr_1.len() as cl_int;
        let mut result = [0.0f32; 1];

        let mut arr_1_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_1.len(),
                arr_1.as_ptr() as *mut c_void,
            )?
        };

        let mut arr_2_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                arr_1.len(),
                arr_1.as_ptr() as *mut c_void,
            )?
        };

        let result_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                std::mem::size_of::<f32>(),
                ptr::null_mut(),
            )?
        };

        let shared_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                arr_1.len(),
                ptr::null_mut(),
            )?
        };

        let _arr_1_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_1_buf, CL_NON_BLOCKING, 0, &arr_1, &[])?
        };
        let _arr_2_buf_write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut arr_2_buf, CL_NON_BLOCKING, 0, &arr_2, &[])?
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&arr_1_buf)
                .set_arg(&arr_2_buf)
                .set_arg(&result_buf)
                .set_arg(&shared_buf)
                .set_arg(&arr_sz)
                .set_global_work_size(arr_1.len())
                .set_wait_event(&_arr_1_buf_write_event)
                .set_wait_event(&_arr_2_buf_write_event)
                .enqueue_nd_range(&self.queue)?
        };

        let mut events: Vec<cl_event> = Vec::default();
        events.push(kernel_event.get());

        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&result_buf, CL_NON_BLOCKING, 0, &mut result, &events)?
        };

        read_event.wait()?;
        Ok(result[0])
    }
}
