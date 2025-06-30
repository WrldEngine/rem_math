use rem_math::native::sum_arr_int32;
use rem_math::native::sum_two_floats32;

#[test]
fn test_sum_arr_i32() {
    let arr = vec![1; 10];
    assert_eq!(10, sum_arr_int32(arr.clone(), false));
    assert_eq!(10, sum_arr_int32(arr.clone(), true));
}

#[test]
fn test_sum_two_floats32() {
    let arr = vec![1.0; 5];
    let expected_arr = vec![2.0, 2.0, 2.0, 2.0, 2.0];

    assert_eq!(expected_arr, sum_two_floats32(arr.clone(), arr.clone(), false));
    assert_eq!(expected_arr, sum_two_floats32(arr.clone(), arr.clone(), true));
}