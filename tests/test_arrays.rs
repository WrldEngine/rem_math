use rem_math::native::multiply_two_ints32;
use rem_math::native::sum_arr_int32;
use rem_math::native::sum_two_floats32;
use rem_math::native::sum_two_ints32;

#[test]
fn test_sum_arr_i32() {
    let arr = [1; 10];
    assert_eq!(10, sum_arr_int32(&arr, false));
    assert_eq!(10, sum_arr_int32(&arr, true));
}

#[test]
fn test_sum_two_floats32() {
    let arr: [f32; 5] = [1.0; 5];
    let expected_arr = vec![2.0; 5];

    assert_eq!(expected_arr, sum_two_floats32(&arr, &arr, ""));
    assert_eq!(expected_arr, sum_two_floats32(&arr, &arr, "simd"));
    assert_eq!(expected_arr, sum_two_floats32(&arr, &arr, "threading"));
}

#[test]
fn test_sum_two_ints32() {
    let arr = [1; 5];
    let expected_arr = vec![2; 5];

    assert_eq!(expected_arr, sum_two_ints32(&arr, &arr, false));
    assert_eq!(expected_arr, sum_two_ints32(&arr, &arr, true));
}

#[test]
fn test_mul_two_ints32() {
    let arr = [2; 5];
    let expected_arr = vec![4; 5];

    assert_eq!(expected_arr, multiply_two_ints32(&arr, &arr, "simd"));
    assert_eq!(expected_arr, multiply_two_ints32(&arr, &arr, "threading"));
    assert_eq!(expected_arr, multiply_two_ints32(&arr, &arr, ""));
}
