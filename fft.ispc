typedef float f32;
typedef double f64;

typedef unsigned int8 u8;
typedef int i32;
typedef unsigned int u32;
typedef unsigned int64 u64;

// #define DOUBLE_PRECISION

#if defined(DOUBLE_PRECISION)
typedef double fxx;
#else
typedef float fxx;
#endif

const uniform fxx pi = 3.14159265358979311599796346854418516159057617187500d;
const uniform fxx tau = 6.28318530717958623199592693708837032318115234375000d;

enum
{
    BaseSize = 16,
    SOAWidth = programCount,
    BaseSizeSoa = 16 / programCount
};

// FFT_RECURSIVE is a simple depth-first implementation
#define FFT_RECURSIVE 0

static inline i32 log2(i32 n)
{
    return 31 - count_leading_zeros(n);
}

static inline uniform i32 log2(uniform i32 n)
{
    return 31 - count_leading_zeros(n);
}

/*
 * 1 / n where n is a power of 2. we compute it a lot and its easy
 */
static inline uniform fxx rcp2(uniform i32 n)
{
#if defined(DOUBLE_PRECISION)
    u64 bits = ((u64)(-log2(n) + 1023)) << 52;
    uniform fxx result = *((fxx *)&bits);
#else
    uniform fxx result = floatbits(((u32)(-log2(n) + 127)) << 23);
#endif

    return result;
}

struct Complex
{
    fxx real;
    fxx imaginary;
};

struct Polar
{
    fxx mag;
    fxx theta;
};

struct SOAComplex
{
    fxx real[SOAWidth];
    fxx imaginary[SOAWidth];
};

static inline Complex vComplex(fxx real, fxx imaginary)
{
    Complex result;
    result.real = real;
    result.imaginary = imaginary;
    return result;
}

static inline uniform Complex uComplex(uniform fxx real, uniform fxx imaginary)
{
    uniform Complex result;
    result.real = real;
    result.imaginary = imaginary;
    return result;
}

static inline Complex operator+(Complex a, Complex b)
{
    Complex result;
    result.real = a.real + b.real;
    result.imaginary = a.imaginary + b.imaginary;
    return result;
}

static inline uniform Complex operator+(uniform Complex a, uniform Complex b)
{
    uniform Complex result;
    result.real = a.real + b.real;
    result.imaginary = a.imaginary + b.imaginary;
    return result;
}

static inline Complex operator-(Complex a, Complex b)
{
    Complex result;
    result.real = a.real - b.real;
    result.imaginary = a.imaginary - b.imaginary;
    return result;
}

static inline uniform Complex operator-(uniform Complex a, uniform Complex b)
{
    uniform Complex result;
    result.real = a.real - b.real;
    result.imaginary = a.imaginary - b.imaginary;
    return result;
}

static inline Complex operator*(Complex a, Complex b)
{
    Complex result;
    result.real = (a.real * b.real) - (a.imaginary * b.imaginary);
    result.imaginary = (a.real * b.imaginary) + (a.imaginary * b.real);
    return result;
}

static inline uniform Complex operator*(uniform Complex a, uniform Complex b)
{
    uniform Complex result;
    result.real = (a.real * b.real) - (a.imaginary * b.imaginary);
    result.imaginary = (a.real * b.imaginary) + (a.imaginary * b.real);
    return result;
}

static inline void sincos64(f64 theta, varying f64 *uniform sinOut, varying f64 *uniform cosOut)
{
    // This is the ispc library sincos function, given new and more polynomial
    // coefficients for 64 bit precision. As far as I can tell it's two
    // odd/even minimax polynomials over (0, pi/2) evaluated using horner's
    // method

    const f64 pi_over_two_vec = pi / 2.;
    const f64 two_over_pi_vec = 2. / pi;
    f64 scaled = theta * two_over_pi_vec;
    f64 k_real = floor(scaled);
    int k = (int)k_real;

    // Reduced range version of x
    f64 x = theta - k_real * pi_over_two_vec;
    int k_mod4 = k & 3;
    bool cos_usecos = (k_mod4 == 0 || k_mod4 == 2);
    bool sin_usecos = (k_mod4 == 1 || k_mod4 == 3);
    bool sin_flipsign = (k_mod4 > 1);
    bool cos_flipsign = (k_mod4 == 1 || k_mod4 == 2);

    const f64 one_vec = 1.;

    // NOTE: You __HAVE__ to put the 'd' postfix on double constants! Also,
    // the ispc parser can't parse scientific notation with doubles--so you
    // need convert e.g 1.5e-2d to 0.015d

    const f64 sin_c2 = -0.1666666666666666666252621249014565907279d;
    const f64 sin_c4 = 0.008333333333333332806754653004915343717313d;
    const f64 sin_c6 = -0.0001984126984126959196882708822010439028941d;
    const f64 sin_c8 = 0.000002755731922392541426937253940148306571934d;
    const f64 sin_c10 = -0.00000002505210837690652036612901208078434685513d;
    const f64 sin_c12 = 0.0000000001605904309514996997761845492036050226018d;
    const f64 sin_c14 = -0.0000000000007647123466459114077851942535220343138305d;
    const f64 sin_c16 = 0.000000000000002810125443051100291973343823470593565709d;
    const f64 sin_c18 = -0.000000000000000007974744329615897771898361082884343947597d;

    const f64 cos_c2 = -0.4999999999999999987531120047042817716558d;
    const f64 cos_c4 = 0.04166666666666665009681327745042733242497d;
    const f64 cos_c6 = -0.001388888888888811370846917197381129181726d;
    const f64 cos_c8 = 0.00002480158730140630424841671041490381481435d;
    const f64 cos_c10 = -0.0000002755731919972249333784864606600704322583d;
    const f64 cos_c12 = 0.000000002087675499859457940122714850177463801631d;
    const f64 cos_c14 = -0.00000000001147064401641941499052272670771500411409d;
    const f64 cos_c16 = 0.00000000000004776320159479282545281379360120129938876d;
    const f64 cos_c18 = -0.0000000000000001507121934943431184262642941395002432378d;

    f64 x2 = x * x;

    f64 sin_formula = x2 * sin_c18 + sin_c16;
    f64 cos_formula = x2 * cos_c18 + cos_c16;

    sin_formula = x2 * sin_formula + sin_c14;
    cos_formula = x2 * cos_formula + cos_c14;

    sin_formula = x2 * sin_formula + sin_c12;
    cos_formula = x2 * cos_formula + cos_c12;

    sin_formula = x2 * sin_formula + sin_c10;
    cos_formula = x2 * cos_formula + cos_c10;

    sin_formula = x2 * sin_formula + sin_c8;
    cos_formula = x2 * cos_formula + cos_c8;

    sin_formula = x2 * sin_formula + sin_c6;
    cos_formula = x2 * cos_formula + cos_c6;

    sin_formula = x2 * sin_formula + sin_c4;
    cos_formula = x2 * cos_formula + cos_c4;

    sin_formula = x2 * sin_formula + sin_c2;
    cos_formula = x2 * cos_formula + cos_c2;

    sin_formula = x2 * sin_formula + one_vec;
    cos_formula = x2 * cos_formula + one_vec;

    sin_formula *= x;

    *sinOut = sin_usecos ? cos_formula : sin_formula;
    *cosOut = cos_usecos ? cos_formula : sin_formula;

    *sinOut = sin_flipsign ? -*sinOut : *sinOut;
    *cosOut = cos_flipsign ? -*cosOut : *cosOut;
}

static inline Complex expi(fxx theta)
{
    Complex result;

#if defined(DOUBLE_PRECISION)
    sincos64(theta, &result.imaginary, &result.real);
#else
    sincos(theta, &result.imaginary, &result.real);
#endif

    return result;
}

static inline Complex operator+(fxx r, Complex z)
{
    Complex result;
    result.real = r + z.real;
    result.imaginary = z.imaginary;
    return result;
}

static inline uniform Complex operator+(uniform fxx r, uniform Complex z)
{
    uniform Complex result;
    result.real = r + z.real;
    result.imaginary = z.imaginary;
    return result;
}

static inline Complex operator-(fxx r, Complex z)
{
    Complex result;
    result.real = r - z.real;
    result.imaginary = z.imaginary;
    return result;
}

static inline uniform Complex operator-(uniform fxx r, uniform Complex z)
{
    uniform Complex result;
    result.real = r - z.real;
    result.imaginary = z.imaginary;
    return result;
}

static inline Complex operator*(Complex z, fxx r)
{
    Complex result;
    result.real = r * z.real;
    result.imaginary = r * z.imaginary;
    return result;
}

static inline Complex operator*(uniform fxx r, Complex z)
{
    Complex result;
    result.real = r * z.real;
    result.imaginary = r* z.imaginary;
    return result;
}

static inline uniform Complex operator*(uniform Complex z, uniform fxx r)
{
    uniform Complex result;
    result.real = r * z.real;
    result.imaginary = r* z.imaginary;
    return result;
}

static inline Complex twiddleFactor(i32 k, uniform i32 N)
{
    assert(k <= N / 2);

    Complex result = expi(-tau * (k * rcp2(N)));

    return result;
}

static void printComplex(uniform Complex z)
{
    if (z.imaginary >= 0) {
        print("% + %i\n", z.real, z.imaginary);
    } else {
        print("% - %i\n", z.real, z.imaginary * -1.);
    }
}

static void printComplex(Complex z)
{
    foreach_active(i) {
        uniform Complex x;
        x.real = extract(z.real, i);
        x.imaginary = extract(z.imaginary, i);
        printComplex(x);
    }
}

static void printComplexArray(uniform Complex array[], uniform i32 count)
{
    for (uniform i32 i = 0; i < count; i++) {
        printComplex(array[i]);
    }
    print("\n");
}

static inline Complex shuffle(Complex z, i32 permutation)
{
    Complex result;
    result.real = shuffle(z.real, permutation);
    result.imaginary = shuffle(z.imaginary, permutation);
    return result;
}

static inline Complex shuffle(Complex a, Complex b, i32 permutation)
{
    Complex result;
    result.real = shuffle(a.real, b.real, permutation);
    result.imaginary = shuffle(a.imaginary, b.imaginary, permutation);
    return result;
}

static inline Complex conjugate(Complex z)
{
    Complex result = z;
    result.imaginary *= -1.;
    return result;
}

static inline uniform Complex conjugate(uniform Complex z)
{
    uniform Complex result = z;
    result.imaginary *= -1.;
    return result;
}

// Reverse algorthim is rom Knuth by way of hackers delight

static inline uniform u32 reverse(uniform u32 i, uniform u32 count)
{
    uniform u32 N = count_leading_zeros(count) + 1;

    uniform u32 t;
    i = (i << 15) | (i >> 17);
    t = (i ^ (i>>10)) & 0x003F801F;
    i = (t | (t<<10)) ^ i;
    t = (i ^ (i>> 4)) & 0x0E038421;
    i = (t | (t<< 4)) ^ i;
    t = (i ^ (i>> 2)) & 0x22488842;
    i = (t | (t<< 2)) ^ i;
    return i >> N;
}
static inline u32 reverse(u32 i, uniform u32 count)
{
    uniform u32 N = count_leading_zeros(count) + 1;

    u32 t;
    i = (i << 15) | (i >> 17);
    t = (i ^ (i>>10)) & 0x003F801F;
    i = (t | (t<<10)) ^ i;
    t = (i ^ (i>> 4)) & 0x0E038421;
    i = (t | (t<< 4)) ^ i;
    t = (i ^ (i>> 2)) & 0x22488842;
    i = (t | (t<< 2)) ^ i;
    return i >> N;
}

static inline uniform i32 thabit(uniform i32 n)
{
    uniform i32 result = (n) << 1;
    result &= (1 << log2(result));
    result += ((result >> 1) - 1);
    return result;
}

/*
 *  Helpers for 4-wide stuff. Not even a little bit necessary for 8 lane gangs
 */

static inline i32 load4(uniform i32 a, uniform i32 b, uniform i32 c, uniform i32 d)
{
    i32 result = a;
    result = insert(result, 1, b);
    result = insert(result, 2, c);
    result = insert(result, 3, d);
    return result;
}

static inline Complex unpacklo4(Complex a, Complex b)
{
    Complex result = shuffle(a, b, load4(0, 4, 1, 5));
    return result;
}

static inline Complex unpackhi4(Complex a, Complex b)
{
    Complex result = shuffle(a, b, load4(2, 6, 3, 7));
    return result;
}

static inline void transpose4(Complex& a, Complex& b, Complex& c, Complex& d)
{
    Complex t0 = unpacklo4(a, c);
    Complex t1 = unpacklo4(b, d);
    Complex t2 = unpackhi4(a, c);
    Complex t3 = unpackhi4(b, d);

    a = unpacklo4(t0, t1);
    b = unpackhi4(t0, t1);
    c = unpacklo4(t2, t3);
    d = unpackhi4(t2, t3);
}

#if defined(FFT_RECURSIVE) && FFT_RECURSIVE

struct FFTInvertParams
{
    Complex *input, *output;
    i32 size;
    fxx scale;
};

#if 0
void fft_internal(uniform Complex input[], uniform Complex output[], uniform i32 count, uniform i32 stride)
{
    if (count == 2) {
        output[0] = (input[0] + input[stride]) * 0.5;
        output[1] = (input[0] - input[stride]) * 0.5;
    } else {
        uniform i32 subCount = count / 2;
        uniform i32 subStride = stride * 2;
        fft_internal(input, output, subCount, subStride);
        fft_internal(input + stride, output + (count / 2), subCount, subStride);

        foreach(i = 0 ... subCount) {
            Complex t = output[i];
            Complex w = expi(-tau * i / count) * output[i + subCount];
            output[i] = (t + w) * 0.5;
            output[i + subCount] = (t - w) * 0.5;
        }
    }
}

export void fft_real(uniform fxx input[], uniform Complex output[], uniform i32 count)
{
    assert(count >= 2);
    assert((count & (count - 1)) == 0);
    uniform i32 N = count / 2;

    fft_internal((uniform Complex *)input, output, N, 1);

    foreach(k = 0 ... N/2 + 1) {
        Complex W = twiddleFactor(k, count);

        Complex Xk = output[k] * 0.5d;
        Complex Xnk = output[N - k] * 0.5d;

        if (k == 0) {
            Xnk = Xk;
        }

        Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
        Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

        output[k] = (a + (b * W)) * 0.5;

        // k'th twiddle factor to N-k'th
        W.real *= -1.;
        output[N - k] = (conjugate(a) + (conjugate(b) * W)) * 0.5;
    }

    output[N].imaginary = 0;
}
#else

void fft_real_internal(uniform FFTInvertParams params, uniform i32 k, uniform Complex output[], uniform i32 count, uniform i32 stride)
{
    if (count == 1) {
        uniform Complex *uniform input = params.input;
        uniform fxx scale = params.scale;
        uniform i32 N = params.size / 2;

        uniform Complex W = twiddleFactor(k, params.size);

        uniform Complex Xk = input[N-k];
        uniform Complex Xnk = conjugate(input[k]);

        if (k == 0) {
            Xk = Xnk;
        }

        uniform Complex a = ((Xk) - Xnk);
        uniform Complex b = (Xnk + (Xk)) * uComplex(0, 1);

        output[0] = a - (b * W);
    } else {
        uniform i32 subCount = count / 2;
        uniform i32 subStride = stride * 2;
        fft_real_internal(params, k, output, subCount, subStride);
        fft_real_internal(params, k + stride, output + (count / 2), subCount, subStride);

        foreach(i = 0 ... subCount) {
            Complex t = output[i];
            Complex w = twiddleFactor(i, count) * output[i + subCount];
            output[i] = (t + w);
            output[i + subCount] = (t - w);
        }
    }
}

export void fft_real(uniform fxx input[], uniform Complex output[], uniform i32 count)
{
    assert(count >= 1);
    assert((count & (count - 1)) == 0);

    uniform FFTInvertParams params;
    params.input = (uniform Complex *)input;
    params.output = output;
    params.size = count;
    params.scale = rcp2(count);

    fft_real_internal(params, 0, (Complex *)output, count / 2, 1);
}

#endif


void fft_invert_real_internal(uniform FFTInvertParams params, uniform i32 k, uniform Complex output[], uniform i32 count, uniform i32 stride)
{
    if (count == 1) {
        uniform Complex *uniform input = params.input;
        uniform fxx scale = params.scale;
        uniform i32 N = params.size / 2;

        uniform Complex W = twiddleFactor(k, params.size);
#if 1
        uniform Complex Xk = conjugate(input[k]) * scale;
        uniform Complex Xnk = conjugate(input[N - k]) * scale;

        uniform Complex a = uComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
        uniform Complex b = uComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);
#else
        uniform Complex Xk = conjugate(input[k]) * scale;
        uniform Complex Xnk = (input[N - k]) * scale;

        uniform Complex a = (Xk) + Xnk;
        uniform Complex b = (Xnk - (Xk)) * uComplex(0, 1);
#endif
        output[0] = a - (b * W);
    } else {
        uniform i32 subCount = count / 2;
        uniform i32 subStride = stride * 2;
        fft_invert_real_internal(params, k, output, subCount, subStride);
        fft_invert_real_internal(params, k + stride, output + (count / 2), subCount, subStride);

        foreach(i = 0 ... subCount) {
            Complex t = output[i];
            Complex w = twiddleFactor(i, count) * output[i + subCount];
            output[i] = (t + w);
            output[i + subCount] = (t - w);
        }
    }
}

export void fft_real_invert(uniform Complex input[], uniform fxx output[], uniform i32 count)
{
    assert(count >= 1);
    assert((count & (count - 1)) == 0);

    uniform FFTInvertParams params;
    params.input = input;
    params.output = (Complex *)output;
    params.size = count;
    params.scale = rcp2(count);

    fft_invert_real_internal(params, 0, (Complex *)output, count / 2, 1);
}

#endif // FFT_RECURSIVE

#if !defined(FFT_RECURSIVE) || !FFT_RECURSIVE

static inline void fft_16(uniform SOAComplex points[BaseSizeSoa], uniform SOAComplex output[BaseSizeSoa])
{
    if (programCount == 8) {
        // In:  [0, 1, 2, 3, 4, 5, 6,  7],  [8, 9, 10, 11, 12, 13, 14, 15]
        // Out: [0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6,  7,  12, 13, 14, 15]
        uniform i32 halfGang = programCount / 2;
        i32 lowPermutation = (programIndex < halfGang) ? programIndex : (programIndex + halfGang);
        i32 highPermutation = (programIndex < halfGang) ? (programIndex + (halfGang * 3)) : programIndex;

        Complex a = vComplex(points[0].real[programIndex], points[0].imaginary[programIndex]);
        Complex b = vComplex(points[1].real[programIndex], points[1].imaginary[programIndex]);

        // 4x4
        Complex c = a + b;
        Complex d = a - b;

        if (programIndex >= 4) {
            d = vComplex(d.imaginary, d.real * -1.);
        }

        Complex e = shuffle(c, d, lowPermutation);
        Complex f = shuffle(d, c, highPermutation);

        Complex g = e + f;
        Complex h = e - f;

        // In:  [0, 1, 2, 3, 4, 5, 6, 7, 8 ] ...
        // Out: [0, 4, 8, 12, 1, 5, 9, 13 ] ...
        i32 kPermutation = (programIndex / 4) + ((programIndex & 0b11) * 4);
        i32 lPermutation = ((programIndex + 8) / 4) + ((programIndex & 0b11) * 4);

        // 2x8
        Complex k = shuffle(g, h, kPermutation);
        Complex l = shuffle(g, h, lPermutation);

        Complex w = twiddleFactor(programIndex, BaseSize);

        // The even twiddle factors of a 16 point dft are the
        // twiddle factors of an 8 point dft
        Complex wL = l * shuffle(w, (programIndex * 2) & 7);

        Complex m = k + wL;
        Complex n = k - wL;

        Complex o = shuffle(m, n, lowPermutation);
        Complex p = shuffle(n, m, highPermutation);

        // 1x16
        Complex wP = p * w;

        Complex q = o + wP;
        Complex r = o - wP;

        // With soa<8>, this is output[programIndex], output[programIndex + 8]
        output[0].real[programIndex] = q.real;
        output[0].imaginary[programIndex] = q.imaginary;
        output[1].real[programIndex] = r.real;
        output[1].imaginary[programIndex] = r.imaginary;
    } else if (programCount == 4) {
        Complex a = vComplex(points[0].real[programIndex], points[0].imaginary[programIndex]);
        Complex b = vComplex(points[1].real[programIndex], points[1].imaginary[programIndex]);
        Complex c = vComplex(points[2].real[programIndex], points[2].imaginary[programIndex]);
        Complex d = vComplex(points[3].real[programIndex], points[3].imaginary[programIndex]);

        // 4x4 point dfts
        Complex e = a + c;
        Complex f = b + d;
        Complex g = a - c;
        Complex h = vComplex(b.imaginary - d.imaginary, (b.real - d.real) * -1.);

        Complex i = e + f;
        Complex j = g + h;
        Complex k = e - f;
        Complex l = g - h;

        transpose4(i, j, k, l);

        // 2x8
        Complex w8 = twiddleFactor(programIndex, BaseSize / 2);

        Complex wK = k * w8;
        Complex wL = l * w8;

        Complex m = i + wK;
        Complex n = j + wL;
        Complex o = i - wK;
        Complex p = j - wL;

        Complex w16odd = twiddleFactor((programIndex * 2) + 1, BaseSize);

        Complex w16lo = shuffle(w8, w16odd, load4(0, 4, 1, 5));
        Complex w16hi = shuffle(w8, w16odd, load4(2, 6, 3, 7));

        // 1x16 point dft
        Complex wN = n * w16lo;
        Complex wP = p * w16hi;

        Complex q = m + wN;
        Complex r = o + wP;
        Complex s = m - wN;
        Complex t = o - wP;

        output[0].real[programIndex] = q.real;
        output[0].imaginary[programIndex] = q.imaginary;
        output[1].real[programIndex] = r.real;
        output[1].imaginary[programIndex] = r.imaginary;
        output[2].real[programIndex] = s.real;
        output[2].imaginary[programIndex] = s.imaginary;
        output[3].real[programIndex] = t.real;
        output[3].imaginary[programIndex] = t.imaginary;
    }
}

static inline void writeAosToSoaComplex(Complex z, i32 i, uniform SOAComplex output[])
{
    /*
        ISPC complains about scattering when we write out SOA data into
        the AOS output. However, each scatter goes into a region of memory
        taken up by another active lane, rather than across gangs. So
        we shuffle our registers from (re, re, ...)(im, im, ...) to
        (re, im, ...)(re, im, ...) and ispc can do packed stores

        8 lane case:
            real:      0  1  2  3  4  5  6  7
            imaginary: 8  9  10 11 12 13 14 15

            to (real, imaginary):
            0  8 1  9 2 10 3 11
            4 12 5 13 6 14 7 15

            This is a right one bit rotate.

        This is, like, 5% faster!
    */

    u32 pc2 = programCount * 2;
    u32 rotate_shift = log2(pc2) - 1;
    u32 mask = pc2 - 1;
    u32 index_imag = programIndex + programCount;

    u32 realPermutation = ((programIndex >> 1) | (programIndex << (rotate_shift))) & mask;
    u32 imaginaryPermutation = ((index_imag >> 1) | (index_imag << (rotate_shift))) & mask;

    Complex x;
    x.real = shuffle(z.real, z.imaginary, realPermutation);
    x.imaginary = shuffle(z.real, z.imaginary, imaginaryPermutation);

    output[i].real[programIndex] =  x.real;
    output[i].imaginary[programIndex] = x.imaginary;
}

static inline void combine_ffts(uniform SOAComplex points[], uniform i32 count)
{
    uniform Complex *uniform pointsAsComplex = (Complex *)points;
    for (uniform i32 subStride = (count / 2) / BaseSize, subCount = BaseSize * 2;
        subCount < count;
        subStride /= 2, subCount *= 2)
    {
        assert(subStride * subCount == count);

        for (uniform i32 k = 0; k < subStride; k++) {
            assert((subCount * k) < count);
            uniform SOAComplex *uniform subPoints = (SOAComplex *)(pointsAsComplex + (subCount * k));

            uniform i32 half = subCount / 2;
            for (uniform i32 i = 0; i < (half / SOAWidth); i++) {
                i32 j = programIndex;

                uniform i32 bOffset = half / SOAWidth;

                Complex a = vComplex(subPoints[i].real[j], subPoints[i].imaginary[j]);
                Complex b = vComplex(subPoints[i + bOffset].real[j], subPoints[i + bOffset].imaginary[j]);
                Complex w = b * twiddleFactor((i * SOAWidth) + j, subCount);

                Complex m = a + w;
                Complex n = a - w;

                subPoints[i].real[j] = m.real;
                subPoints[i].imaginary[j] = m.imaginary;
                subPoints[i + bOffset].real[j] = n.real;
                subPoints[i + bOffset].imaginary[j] = n.imaginary;
            }
        }
    }
}

export void fft_real(uniform fxx input[], uniform Complex output[], uniform i32 count)
{
    assert(count >= BaseSize);
    assert((count & (count - 1)) == 0);
    assert(((ptrdiff_t)output & 31) == 0);

    uniform SOAComplex *uniform soaOutput = (SOAComplex *)output;

    uniform i32 N = (count & (~(count - 1))) / 2;
    uniform i32 stride = (N / BaseSize);

    for (uniform i32 k = 0; k < stride; k++) {
        uniform SOAComplex dft4[BaseSizeSoa];

        uniform Complex *uniform subInput = ((uniform Complex *)input) + k;
        uniform SOAComplex *uniform subOutput = (SOAComplex *)(output + (BaseSize * reverse(k, stride)));

        for (uniform i32 i = 0; i < BaseSizeSoa; i++) {
            i32 j = programIndex;

            dft4[i].real[j] = subInput[(i * SOAWidth * stride) + (j * stride)].real;
            dft4[i].imaginary[j] = subInput[(i * SOAWidth * stride) + (j * stride)].imaginary;
        }

        fft_16(dft4, subOutput);
    }

    combine_ffts(soaOutput, N);

    uniform i32 subCount = N / 2;
    for (uniform i32 i = 0; i < subCount / SOAWidth; i++) {
        i32 j = programIndex;

        uniform i32 bOffset = subCount / SOAWidth;

        Complex a = vComplex(soaOutput[i].real[j], soaOutput[i].imaginary[j]);
        Complex b = vComplex(soaOutput[i + bOffset].real[j], soaOutput[i + bOffset].imaginary[j]);

        Complex w = b * twiddleFactor((i * SOAWidth) + j, N);

        Complex m = a + w;
        Complex n = a - w;

        writeAosToSoaComplex(m, i, soaOutput);
        writeAosToSoaComplex(n, i + bOffset, soaOutput);
    }

    // Set the lanemask to exclude the first lane and do the first iteration
    // out of the loop. If you do foreach(i = 1 ... N) ispc generates a sse2
    // vector load on unaligned memory with the force-aligned-memory option
    // set. boo
    if (programIndex != 0) {
        i32 k = programIndex;
        Complex W = twiddleFactor(k, count);

        Complex Xk = output[k];
        Complex Xnk = output[N - k];

        Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
        Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

        output[k] = (a + (b * W)) * 0.5;

        // k'th twiddle factor to N-k'th
        W.real *= -1.;
        output[N - k] = (conjugate(a) + (conjugate(b) * W)) * 0.5;
    }

    foreach(k = programCount ... N/2) {
        Complex W = twiddleFactor(k, count);

        Complex Xk = output[k];
        Complex Xnk = output[N - k];

        Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
        Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

        output[k] = (a + (b * W)) * 0.5;

        // k'th twiddle factor to N-k'th
        W.real *= -1.;
        output[N - k] = (conjugate(a) + (conjugate(b) * W)) * 0.5;
    }

    // k == 0, N/2+1
    {
        uniform Complex x = output[0];
        output[0] = uComplex(x.real + x.imaginary, 0);
        output[N] = uComplex(x.real - x.imaginary, 0);
        output[N / 2].imaginary *= -1.;
    }
}

export void fft_real_invert(uniform Complex input[], uniform fxx output[], uniform i32 count)
{
    assert(count >= BaseSize);
    assert((count & (count - 1)) == 0);
    assert(((ptrdiff_t)output & 31) == 0);

    uniform Complex *uniform outputAsComplex = (Complex *)output;
    uniform SOAComplex *uniform soaOutput = (SOAComplex *)output;

    uniform i32 N = (count & (~(count - 1))) / 2;
    uniform i32 stride = N / BaseSize;
    uniform fxx scale = rcp2(count);

    // k == 0. There's symmetry, but with weird off by one alignment, because of the value at nyquist
    {
        uniform SOAComplex dft4[BaseSizeSoa];

        for (uniform i32 i = 0; i < BaseSizeSoa; i++) {
            i32 j = programIndex;
            i32 idx = (i * SOAWidth * stride) + (j * stride);

            Complex Xk = conjugate(input[idx]) * scale;
            Complex Xnk = conjugate(input[N - idx]) * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;
        }

        fft_16(dft4, soaOutput);
    }

    // k == N / 2. Symmetry is within one k block
    {
        uniform SOAComplex dft4[BaseSizeSoa];
        uniform SOAComplex *uniform subOutput = soaOutput + BaseSizeSoa;

        for (uniform i32 i = 0; i < BaseSizeSoa / 2; i++) {
            i32 j = programIndex;
            i32 idx = (stride / 2) + (i * SOAWidth * stride) + (j * stride);

            Complex Xk = conjugate(input[idx]) * scale;
            Complex Xnk = conjugate(input[N - idx]) * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;

            Complex y = conjugate(a - (b * (W * -1.)));
            y = shuffle(y, (programCount - 1) - j);

            dft4[(BaseSizeSoa - 1) - i].real[j] = y.real;
            dft4[(BaseSizeSoa - 1) - i].imaginary[j] = y.imaginary;
        }

        fft_16(dft4, subOutput);
    }

    // Symmetry between k and N - k
    for (uniform i32 k = 1; k < (stride / 2); k++) {
        uniform SOAComplex dft4[BaseSizeSoa];
        uniform SOAComplex dft4Mirror[BaseSizeSoa];

        i32 kVec = k;
        kVec = insert(kVec, 1, N-k);
        i32 offsets = BaseSize * reverse(kVec, stride);

        uniform i32 outputOffset = extract(offsets, 0);
        uniform i32 mirrorOffset = extract(offsets, 1);

        uniform SOAComplex *uniform subOutput = (SOAComplex *)(outputAsComplex + outputOffset);
        uniform SOAComplex *uniform subOutputMirror = (SOAComplex *)(outputAsComplex + mirrorOffset);

        for (uniform i32 i = 0; i < BaseSizeSoa; i++) {
            i32 j = programIndex;

            i32 idx = k + (i * SOAWidth * stride) + (j * stride);
            assert(idx <= N);

            Complex Xk = conjugate(input[idx]) * scale;
            Complex Xnk = conjugate(input[N - idx]) * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;

            Complex y = conjugate(a - (b * (W * -1.)));
            y = shuffle(y, (programCount - 1) - j);

            dft4Mirror[(BaseSizeSoa - 1) - i].real[j] = y.real;
            dft4Mirror[(BaseSizeSoa - 1) - i].imaginary[j] = y.imaginary;
        }

        fft_16(dft4, subOutput);
        fft_16(dft4Mirror, subOutputMirror);
    }

    combine_ffts(soaOutput, N);

    uniform i32 subCount = N / 2;
    for (uniform i32 i = 0; i < subCount / SOAWidth; i++) {
        i32 j = programIndex;

        uniform i32 bOffset = subCount / SOAWidth;

        Complex a = vComplex(soaOutput[i].real[j], soaOutput[i].imaginary[j]);
        Complex b = vComplex(soaOutput[i + bOffset].real[j], soaOutput[i + bOffset].imaginary[j]);

        Complex w = b * twiddleFactor((i * SOAWidth) + j, N);

        Complex m = a + w;
        Complex n = a - w;

        writeAosToSoaComplex(m, i, soaOutput);
        writeAosToSoaComplex(n, i + bOffset, soaOutput);
    }
}

export void fft_real_invert_polar(uniform Polar input[], uniform fxx output[], uniform i32 count)
{
    assert(count >= BaseSize);
    assert((count & (count - 1)) == 0);
    assert(((ptrdiff_t)output & 31) == 0);

    uniform Complex *uniform outputAsComplex = (Complex *)output;
    uniform SOAComplex *uniform soaOutput = (SOAComplex *)output;

    uniform i32 N = (count) / 2;
    uniform i32 stride = N / BaseSize;
    uniform fxx scale = rcp2(count);

    // k == 0. There's symmetry, but with weird off by one alignment, because of the value at nyquist
    {
        uniform SOAComplex dft4[BaseSizeSoa];

        for (uniform i32 i = 0; i < BaseSizeSoa; i++) {
            i32 j = programIndex;
            i32 idx = (i * SOAWidth * stride) + (j * stride);

            Polar P_Xk = input[idx];
            Polar P_Xnk = input[N - idx];

            Complex Xk, Xnk;
            sincos(P_Xk.theta, &Xk.imaginary, &Xk.real);
            sincos(P_Xnk.theta, &Xnk.imaginary, &Xnk.real);

            Xk = conjugate(Xk) * P_Xk.mag * scale;
            Xnk = conjugate(Xnk) * P_Xnk.mag * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;
        }

        fft_16(dft4, soaOutput);
    }

    // k == N / 2. Symmetry is within one k block
    {
        uniform SOAComplex dft4[BaseSizeSoa];
        uniform SOAComplex *uniform subOutput = soaOutput + BaseSizeSoa;

        for (uniform i32 i = 0; i < BaseSizeSoa / 2; i++) {
            i32 j = programIndex;
            i32 idx = (stride / 2) + (i * SOAWidth * stride) + (j * stride);

            Polar P_Xk = input[idx];
            Polar P_Xnk = input[N - idx];

            Complex Xk, Xnk;
            sincos(P_Xk.theta, &Xk.imaginary, &Xk.real);
            sincos(P_Xnk.theta, &Xnk.imaginary, &Xnk.real);

            Xk = conjugate(Xk) * P_Xk.mag * scale;
            Xnk = conjugate(Xnk) * P_Xnk.mag * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;

            Complex y = conjugate(a - (b * (W * -1.)));
            y = shuffle(y, (programCount - 1) - j);

            dft4[(BaseSizeSoa - 1) - i].real[j] = y.real;
            dft4[(BaseSizeSoa - 1) - i].imaginary[j] = y.imaginary;
        }

        fft_16(dft4, subOutput);
    }

    // Symmetry between k and N - k
    for (uniform i32 k = 1; k < (stride / 2); k++) {
        uniform SOAComplex dft4[BaseSizeSoa];
        uniform SOAComplex dft4Mirror[BaseSizeSoa];

        i32 kVec = k;
        kVec = insert(kVec, 1, N-k);
        i32 offsets = BaseSize * reverse(kVec, stride);

        uniform i32 outputOffset = extract(offsets, 0);
        uniform i32 mirrorOffset = extract(offsets, 1);

        uniform SOAComplex *uniform subOutput = (SOAComplex *)(outputAsComplex + outputOffset);
        uniform SOAComplex *uniform subOutputMirror = (SOAComplex *)(outputAsComplex + mirrorOffset);

        for (uniform i32 i = 0; i < BaseSizeSoa; i++) {
            i32 j = programIndex;

            i32 idx = k + (i * SOAWidth * stride) + (j * stride);
            assert(idx <= N);

            Polar P_Xk = input[idx];
            Polar P_Xnk = input[N - idx];

            Complex Xk, Xnk;
            sincos(P_Xk.theta, &Xk.imaginary, &Xk.real);
            sincos(P_Xnk.theta, &Xnk.imaginary, &Xnk.real);

            Xk = conjugate(Xk) * P_Xk.mag * scale;
            Xnk = conjugate(Xnk) * P_Xnk.mag * scale;
            Complex W = twiddleFactor(idx, count);

            Complex a = vComplex(Xk.real + Xnk.real, Xk.imaginary - Xnk.imaginary);
            Complex b = vComplex(Xk.imaginary + Xnk.imaginary, Xnk.real - Xk.real);

            Complex x = a - (b * W);
            dft4[i].real[j] = x.real;
            dft4[i].imaginary[j] = x.imaginary;

            Complex y = conjugate(a - (b * (W * -1.)));
            y = shuffle(y, (programCount - 1) - j);

            dft4Mirror[(BaseSizeSoa - 1) - i].real[j] = y.real;
            dft4Mirror[(BaseSizeSoa - 1) - i].imaginary[j] = y.imaginary;
        }

        fft_16(dft4, subOutput);
        fft_16(dft4Mirror, subOutputMirror);
    }

    combine_ffts(soaOutput, N);

    uniform i32 subCount = N / 2;
    for (uniform i32 i = 0; i < subCount / SOAWidth; i++) {
        i32 j = programIndex;

        uniform i32 bOffset = subCount / SOAWidth;

        Complex a = vComplex(soaOutput[i].real[j], soaOutput[i].imaginary[j]);
        Complex b = vComplex(soaOutput[i + bOffset].real[j], soaOutput[i + bOffset].imaginary[j]);

        Complex w = b * twiddleFactor((i * SOAWidth) + j, N);

        Complex m = a + w;
        Complex n = a - w;

        writeAosToSoaComplex(m, i, soaOutput);
        writeAosToSoaComplex(n, i + bOffset, soaOutput);
    }
}
#endif
