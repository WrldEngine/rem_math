/*
 * Experimental module.
 *
 * This code is under active development and provided for internal testing only.
 * APIs, behavior, and implementation details are unstable and may change
 * without notice. Do not rely on this functionality in production or
 * draw conclusions from current results.
*/

class RemMath {
    private static native int[] sum_two_ints32(int[] a, int[] b, String method);
    static {
        System.loadLibrary("rem_math");
    }

    public static void main(String[] args) {
        int[] a = {1, 2, 3};
        int[] b = {1, 2, 3};

        int[] output = RemMath.sum_two_ints32(a, b, "gpu");
        System.out.println(output);
    }
}