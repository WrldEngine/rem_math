use std::env;

fn main() {
    if cfg!(windows) {
        // Note: in windows should be installed OpenCL.lib file and other additional, before building
        // make sure it is already installed
        // Installation via vcpkg is recommended, run `vcpkg install opencl`
        let open_cl_lib_path = env::var("OPEN_CL_LIB_PATH")
            .unwrap_or("C:\\Users\\user\\vcpkg\\installed\\x64-windows\\lib".into());
        println!("cargo:rustc-link-search=native={}", open_cl_lib_path);
    }

    println!("cargo:rustc-link-lib=dylib=OpenCL");
}
