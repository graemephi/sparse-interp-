#ifndef MAX_EXTERNAL
#define MAX_EXTERNAL 0
#endif

// Note: uses asserts for error handling.
// Todo: at this point this shared header stuff should be its own file probably

#define _CRT_SECURE_NO_WARNINGS
#include <stdint.h>
#include <stddef.h>
#include <math.h>

typedef size_t usize;
typedef ptrdiff_t isize;
typedef double f64;
typedef float f32;
typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef u32 b32;
typedef u8 b8;

// Just follow numpy's convention for alignment
#define IS_ALIGNED_64(x) ((((size_t)x) & 63ull) == 0)
#define ROUND_UP_64(x) (IS_ALIGNED_64(x) ? (x) : ((x) & ~63ull) + 64ull)
#define IS_POWER_2(x) ((x) != 0 && ((x) & ((x) - 1)) == 0)

enum {
    InterpolationNodes = 25,

    // If InterpolationNodesBilinear*InterpolationNodesBilinear >
    // InterpolationNodes, then the synth crashes when dictionaries are swapped
    // Who knows why
    InterpolationNodesBilinear = 5
};

enum {
    CoefSet_A,
    CoefSet_B,
    CoefSet_C,
    CoefSet_D
};

typedef struct sparse_interp_api {
    void *handle;

    void (*init_api)(struct sparse_interp_api *api);

    isize (*init_size_req)(void *previous_state);
    isize (*init)(void *previous_state, u8 *buf, isize buf_len);

    isize (*new_size_req)(char const *dict_name);
    isize (*new)(char const *dict_name, u8 *buf, isize buf_len);

    isize (*load_dictionary_size_req)(char const *dict_path, char const *presets_path);
    isize (*load_dictionary)(char const *path, char const *presets_path, u8 *buf, isize buf_len);

    void (*note)(void *instance, f32 velocity);

    void (*set_coefficients)(void *instance, i32 coef_set, i32 count, i32 *indices, f32 *values);
    void (*set_all)(void *instance, i32 coef_set, i32 count, f32 *coefs);
    void (*set_preset)(void *instance, i32 coef_set, i32 preset);
    void (*set_t_x)(void *instance, f32 t);
    void (*set_t_y)(void *instance, f32 t);
    void (*set_dictionary)(void *instance, char *dict_name);
    void (*recompute)(void *instance);

    void *max_perform;
} sparse_interp_api;

u64 hash(char const *p, isize len)
{
	usize h = 1099511628211;
    u8 *b = (u8 *)p;
    for (i32 i = 0; i < len; i++) {
        h ^= b[i];
        h *= 0xcbf29ce484222325;
    }
    return h;
}
#define HASH(arr_or_immediate) (hash(arr_or_immediate, sizeof(arr_or_immediate)))

#if MAX_EXTERNAL == 0 || DYNAMIC_LOADING == 0

#pragma warning(disable: 4204)

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#ifndef true
#define true 1
#endif

#ifndef false
#define false 0
#endif

// Why does windows define this anyway???
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#include "fft_ispc.h"
typedef struct Polar Polar;

#define internal static

#define Pi 3.14159265358979323846f
#define Tau 6.28318530717958647692f
#define PiOver2 1.57079632679489661923f

// t smoothing coefficients
#define T_A 0.99973823487867019777f
#define T_B (1 - T_A)

typedef struct int2
{
    i32 x, y;
} int2;

typedef struct float2
{
    f32 x, y;
} float2;

typedef struct Word
{
    float *mid;
    float *side;
} Word;

enum DictType
{
    DictType_Time,
    DictType_Polar,
    DictTypeCount
};

typedef struct Dict
{
    char const *name;

    i32 preset_count;
    f32 *presets;

    u32 type;
    i32 word_len;
    i32 sample_len;
    i32 word_count;
    int2 buf_dim;
    f32 *buf;
} Dict;

typedef struct Npy
{
    b8 is_float32;
    b8 is_fortran_order;
    int2 shape;
} Npy;

typedef struct State
{
    // Hash table.
    i32 dicts_len;
    i32 dicts_cap;
    Dict **dicts;
} State;

typedef struct Instance
{
    Dict *dict;

    f32 t_x;
    f32 t_x_z;
    f32 t_y;
    f32 t_y_z;
    f32 *a_coefficients;
    f32 *b_coefficients;
    f32 *c_coefficients;
    f32 *d_coefficients;
    b32 bilinear;

    f32 active;

    f32 *scratch[2];

    f32 *sample_bufs[2];
    f32 *sample_end;
    u32 sample_index;
    b32 needs_recompute;
} Instance;

typedef struct Stack
{
    u8 *buf;
    u8 *ptr;
    u8 *end;
} Stack;

#define stack_push_n(stack, type, count) (assert(count >= 0), assert(count == 0 || (sizeof(type) * (usize)(count)) >= sizeof(type)), assert(count == 0 || (sizeof(type) * (usize)(count)) >= (usize)(count)), stack_push(stack, sizeof(type) * (usize)(count)))
#define stack_push_one(stack, type) (assert(sizeof(type) <= sizeof(stack_dryrun_buf)), stack_push_zero(stack, sizeof(type)))

#define DRY_RUN 0
u8 stack_dryrun_buf[1024];

static Dict empty_dict = {0};

internal void *stack_push_dryrun(Stack *stack, isize size)
{
    memset(stack_dryrun_buf, 0, sizeof(stack_dryrun_buf));
    stack->ptr += ROUND_UP_64(size);
    return stack_dryrun_buf;
}

internal void *stack_push(Stack *stack, isize size)
{
    if (stack->buf == DRY_RUN) {
        return stack_push_dryrun(stack, size);
    }

    if (stack->ptr + size > stack->end) {
        assert(!"Stack overflow");
        return 0;
    }

    u8 *ptr = stack->ptr;
    stack->ptr += ROUND_UP_64(size);
    return ptr;
}

internal void *stack_push_zero(Stack *stack, isize size)
{
    u8 *ptr = stack_push(stack, size);
    if (ptr && stack->buf != DRY_RUN) {
        memset(ptr, 0, size);
    }
    return ptr;
}

internal void *stack_push_fread(Stack *stack, isize elem_size, isize elem_count, FILE *f)
{
    isize size = elem_size * elem_count;
    u8 *ptr = stack_push(stack, size);

    if (ptr && ptr != stack_dryrun_buf) {
        isize nread = fread(ptr, elem_size, elem_count, f);
        if (nread != elem_count) {
            // Todo (robustness): reset file cursor and stack pointer
            ptr = 0;
        }
    }

    return ptr;
}

static State *state = 0;

internal char const *scan_to(char const *p, isize p_len, char const *s, isize s_len)
{
    char const *result = 0;

    char const *p_start = p;
    char const *p_end = p + p_len;
    char const *s_last = s + s_len - 1;

    for (;;) {
        while (*s_last != *p && p != p_end) {
            p++;
        }

        if ((p - p_start) < s_len) {
            p++;
            continue;
        }

        if (p == p_end) {
            break;
        }

        char const *ss = s_last;
        char const *pp = p;
        isize result_len = 0;
        assert(*ss == *pp);
        while (*ss == *pp) {
            ss--;
            pp--;
            result_len++;
        }

        if (result_len == s_len) {
            result = pp + 1;
            break;
        }

        p++;
    }

    return result;
}

internal char const *consume(char const *p, char const *s)
{
    while (*p == *s && *s != '\0') {
        p++;
        s++;
    }

    if (*s != '\0') {
        return 0;
    }

    return p;
}

internal char const *consume_to(char const *p, isize p_len, char const *s, isize s_len)
{
    p = scan_to(p, p_len, s, s_len);

    if (!p || p == (p + p_len)) {
        return 0;
    }

    p = consume(p, s);
    return p;
}

internal char const *consume_uint(char const *p, i32 *out)
{
    i32 result = 0;

    while (*p >= '0' && *p <= '9') {
        result = (result * 10) + (*p - '0');
        p++;
    }

    if (out) {
        *out = result;
    }

    return p;
}

#define STR_ARG_LEN(s) (s), (sizeof(s) - 1)

internal Npy parse_npy_header(char const *buf, isize len)
{
    Npy result = {0};
    char const *buf_end = buf + len;

    buf = consume(buf, "{'descr': '");

    result.is_float32 = strncmp(buf, STR_ARG_LEN("<f4")) == 0;

    buf = consume_to(buf, buf_end - buf, STR_ARG_LEN("'fortran_order': "));

    result.is_fortran_order = strncmp(buf, STR_ARG_LEN("True")) == 0;

    buf = consume_to(buf, buf_end - buf, STR_ARG_LEN("'shape': ("));
    buf = consume_uint(buf, &result.shape.x);
    buf = consume(buf, ", ");
    buf = consume_uint(buf, &result.shape.y);

    return result;
}

internal Npy read_npy_header(FILE *f)
{
    char preamble[10] = {0};
    fread(preamble, 1, sizeof(preamble), f);

    // numpy magic + major version number
    char const *len_ptr = consume(preamble, "\x93NUMPY\x01");
    assert(len_ptr);

    // skip minor version number
    len_ptr++;

    short header_len = *((short *)len_ptr);

    char header[1024];
    assert(header_len < sizeof(header));

    fread(header, sizeof(char), header_len, f);

    Npy result = parse_npy_header(header, header_len);
    return result;
}

internal void add_dict(Dict *dict)
{
    assert(state);
    assert(dict->name);
    // todo: expand
    assert(state->dicts_len != state->dicts_cap);

    size_t len = strlen(dict->name);
    u64 h = hash(dict->name, len);
    isize idx = h % state->dicts_cap;
    while (state->dicts[idx]) {
        if (state->dicts[idx]->name == dict->name) {
            // name already present. fail silently
            return;
        }
        idx = (idx + 1) % state->dicts_cap;
    }
    state->dicts[idx] = dict;
    state->dicts_len++;
}

internal Dict *get_dict(char const *name)
{
    assert(state);
    Dict *result = 0;

    if (name == 0) {
        return result;
    }

    size_t len = strlen(name);
    u64 h = hash(name, len);
    isize idx = h % state->dicts_cap;
    do {
        result = state->dicts[idx];
        // Names come from max symbols or constants so have pointer equality
        if (result == 0 || result->name == name) {
            break;
        }
        assert((strncmp(result->name, name, len) != 0));

        idx = (idx + 1) % state->dicts_cap;
    } while (1);

    return result;
}

internal isize load_dict(char const *dict_path, char const *presets_path, Stack *stack)
{
    assert(state);

    u8 *stack_start = stack->ptr;

    FILE *f = fopen(dict_path, "rb");
    if (!f) {
        return 0;
    }

    FILE *g = fopen(presets_path, "rb");
    if (!g) {
        fclose(f);
        return 0;
    }

    Npy dict_hdr = read_npy_header(f);
    assert(dict_hdr.is_float32);
    assert(dict_hdr.is_fortran_order == 0);

    isize dict_size = dict_hdr.shape.x * dict_hdr.shape.y;

    u8 *buf = stack_push_fread(stack, sizeof(f32), dict_size, f);
    assert(buf);

    fclose(f);

    i32 *word_start = (i32 *)buf;
    i32 *word_end = word_start + dict_hdr.shape.y;
    i32 word_len = (i32)(word_end - word_start) / 2;
    i32 sample_len = word_len;
    u32 type = DictType_Time;
    // Todo: make the dictionary type explicit rather than implicit
    if (stack->buf != DRY_RUN) {
        i32 *mid_end = word_start + dict_hdr.shape.y / 2 - 1;

        if (*mid_end == -1) {
            type = DictType_Polar;

            while (*mid_end == -1) {
                mid_end--;
            }
            mid_end++;
            word_len = (i32)(mid_end - word_start);
            sample_len = word_len - 2;

            assert(IS_POWER_2(sample_len));
            assert(IS_ALIGNED_64(dict_hdr.shape.y * sizeof(f32)));
        }
    }

    Npy presets_hdr = read_npy_header(g);
    assert(presets_hdr.is_float32);
    assert(presets_hdr.is_fortran_order == 0);
    f32 *presets = stack_push_fread(stack, sizeof(f32), presets_hdr.shape.x * presets_hdr.shape.y, g);
    i32 preset_count = presets_hdr.shape.x;
    fclose(g);

    Dict *dict = stack_push_one(stack, Dict);
    *dict = (Dict) {
        .name = dict_path,
        .buf = (f32 *)buf,
        .buf_dim = (int2) { dict_hdr.shape.y, dict_hdr.shape.x },
        .type = type,
        .word_len = word_len,
        .sample_len = sample_len,
        .word_count = dict_hdr.shape.x,
        .presets = presets,
        .preset_count = preset_count
    };

    // Todo: move this up the call stack somewhere so this code doesn't have to know about dry run stacks
    if (stack->buf != DRY_RUN) {
        add_dict(dict);
    }

    return stack->ptr - stack_start;
}

internal isize init(void *previous_state, u8 *buf, isize buf_len)
{
    assert(IS_ALIGNED_64((size_t)buf));

    Stack *stack = &(Stack) {
        buf,
        buf,
        buf + buf_len
    };

    if (previous_state && buf == 0) {
        state = previous_state;
        return 0;
    }

    state = stack_push_one(stack, State);
    state->dicts_len = 0;
    state->dicts_cap = 256;
    state->dicts = stack_push_zero(stack, sizeof(state->dicts[0]) * state->dicts_cap);

    return stack->ptr - stack->buf;
}

internal isize init_size_req(void *previous_state)
{
    return init(previous_state, DRY_RUN, 0);
}

internal isize load_dictionary(char const *dict_path, char const *preset_path, u8 *buf, isize buf_len)
{
    assert(state);

    if (get_dict(dict_path) != 0) {
        return 0;
    }

    Stack *stack = &(Stack) {
        buf,
        buf,
        buf + buf_len
    };

    isize dict_len = load_dict(dict_path, preset_path, stack);

    return dict_len;
}

internal isize load_dictionary_size_req(char const *dict_path, char const *preset_path)
{
    return load_dictionary(dict_path, preset_path, DRY_RUN, 0);
}

internal Word get_word(Dict *dict, isize n)
{
    assert(n < dict->word_count);

    f32 *mid = dict->buf + dict->buf_dim.x * n;
    f32 *side = mid + (dict->buf_dim.x / 2);

    return (Word) { mid, side };
}

internal f32 *get_preset(Dict *dict, i32 i)
{
    i32 preset = (i32)(((u32)i) % dict->preset_count);
    return dict->presets + preset * dict->word_count;
}

internal f32 lerp(f32 a, f32 b, f32 t)
{
    return (a - a*t) + b*t;
}

/* iq */
float inverse_smoothstep( float x )
{
   float a = acosf(1.0f-2.0f*x)/3.0f;
   return (1.0f + sinf(a)*sqrtf(3.0f) - cosf(a))/2.0f;
}

internal f32 smooth(f32 x)
{
    // return x;
    return x * x * (3.f - 2.f * x);
    // return x * x * x * (x * (x * 6.f - 15.f) + 10.f);
    // return inverse_smoothstep(x);
}

internal f32 clamp01(f32 x)
{
    return (x < 0) ? 0
         : (x > 1) ? 1
         : x;
}

internal f32 absf32(f32 x)
{
    return x > 0 ? x : -x;
}

internal void interpolate(Instance *instance)
{
    u32 ftz =_MM_GET_FLUSH_ZERO_MODE();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    instance->needs_recompute = false;

    if (instance->bilinear) {
        // Copy n paste. Figure out better factoring later
        f32 *mid, *side, *left, *right;

        mid = instance->scratch[0];
        side = instance->scratch[1];
        left = instance->sample_bufs[0];
        right = instance->sample_bufs[1];
        Dict *dict = instance->dict;

        for (isize m = 0; m < InterpolationNodesBilinear; m++) {
            f32 tm = smooth((f32)m / (InterpolationNodesBilinear - 1));

            for (isize i = 0; i < InterpolationNodesBilinear; i++) {
                memset(mid, 0, dict->word_len * sizeof(f32));
                memset(side, 0, dict->word_len * sizeof(f32));

                f32 ti = smooth((f32)i / (InterpolationNodesBilinear - 1));
                for (isize j = 0; j < dict->word_count; j++) {
                    f32 coefficient_x1 = lerp(instance->a_coefficients[j], instance->b_coefficients[j], ti);
                    f32 coefficient_x2 = lerp(instance->c_coefficients[j], instance->d_coefficients[j], ti);
                    f32 coefficient = lerp(coefficient_x1, coefficient_x2, tm);
                    if (coefficient) {
                        f32 dist_from_corner = 0.707106781187f - sqrtf((tm - .5f)*(tm - .5f) + (ti - .5f)*(ti - .5f));
                        f32 v = coefficient * ((0.707106781187f * cosf(-PiOver2 + Pi*dist_from_corner)) + 1.f);
                        Word w = get_word(dict, j);
                        for (isize k = 0; k < dict->word_len; k++) {
                            mid[k] += w.mid[k] * v;
                            side[k] += w.side[k] * v;
                        }
                    }
                }

                if (instance->dict->type == DictType_Polar) {
                    fft_real_invert_polar((Polar *)mid, left, dict->sample_len);
                    fft_real_invert_polar((Polar *)side, right, dict->sample_len);

                    for (isize s = 0; s < dict->sample_len; s++) {
                        f32 L = left[s];
                        f32 R = right[s];
                        left[s] = (L + R) * 0.5f;
                        right[s] = (L - R) * 0.5f;
                    }
                } else {
                    for (isize s = 0; s < dict->sample_len; s++) {
                        f32 L = mid[s];
                        f32 R = side[s];
                        left[s] = (L + R) * 0.5f;
                        right[s] = (L - R) * 0.5f;
                    }
                }

                left += dict->sample_len;
                right += dict->sample_len;
            }
        }
    } else {
        f32 *mid, *side, *left, *right;

        mid = instance->scratch[0];
        side = instance->scratch[1];
        left = instance->sample_bufs[0];
        right = instance->sample_bufs[1];
        Dict *dict = instance->dict;

        for (isize i = 0; i < InterpolationNodes; i++) {
            memset(mid, 0, dict->word_len * sizeof(f32));
            memset(side, 0, dict->word_len * sizeof(f32));

            f32 t = smooth((f32)i / (InterpolationNodes - 1));
            for (isize j = 0; j < dict->word_count; j++) {
                f32 coefficient = lerp(instance->a_coefficients[j], instance->b_coefficients[j], t);
                if (coefficient) {
                    f32 v = coefficient * ((0.707106781187f * cosf(-PiOver2 + Pi*t)) + 1.f);
                    Word w = get_word(dict, j);
                    for (isize k = 0; k < dict->word_len; k++) {
                        mid[k] += w.mid[k] * v;
                        side[k] += w.side[k] * v;
                    }
                }
            }

            if (instance->dict->type == DictType_Polar) {
                fft_real_invert_polar((Polar *)mid, left, dict->sample_len);
                fft_real_invert_polar((Polar *)side, right, dict->sample_len);

                for (isize s = 0; s < dict->sample_len; s++) {
                    f32 L = left[s];
                    f32 R = right[s];
                    left[s] = (L + R) * 0.5f;
                    right[s] = (L - R) * 0.5f;
                }
            } else {
                for (isize s = 0; s < dict->sample_len; s++) {
                    f32 L = mid[s];
                    f32 R = side[s];
                    left[s] = (L + R) * 0.5f;
                    right[s] = (L - R) * 0.5f;
                }
            }

            left += dict->sample_len;
            right += dict->sample_len;
        }
    }

    _MM_SET_FLUSH_ZERO_MODE(ftz);
}

internal void note(Instance *instance, f32 velocity)
{
    if (velocity) {
        if (instance->active == 0) {
            instance->sample_index = 0;
        }

        if (instance->needs_recompute) {
            interpolate(instance);
        }
    }
}

internal void copy_coefficients(f32 *coefficients, i32 count, i32 *indices, f32 *values)
{
    i32 *index = indices;
    f32 *value = values;
    for (i32 i = 0; i < count; i++) {
        coefficients[*index] = *value;
        index++;
        value++;
    }
}

internal f32 *get_coefficients(Instance *instance, i32 coef_set)
{
    f32 *result = 0;

    switch (coef_set) {
        case CoefSet_A: {
            result = instance->a_coefficients;
        } break;
        case CoefSet_B: {
            result = instance->b_coefficients;
        } break;
        case CoefSet_C: {
            result = instance->c_coefficients;
        } break;
        case CoefSet_D: {
            result = instance->d_coefficients;
        } break;
    }

    return result;
}

internal void set_coefficients(Instance *instance, i32 coef_set, i32 count, i32 *indices, f32 *values)
{
    f32 *instance_coefficients = get_coefficients(instance, coef_set);

    if (instance_coefficients) {
        if (count <= instance->dict->word_count) {
            copy_coefficients(instance_coefficients, count, indices, values);
            instance->needs_recompute = true;
        }
    }
}

internal void set_all(Instance *instance, i32 coef_set, i32 count, f32 *coefs)
{
    f32 *instance_coefficients = get_coefficients(instance, coef_set);

    if (instance_coefficients) {
        if (count <= instance->dict->word_count) {
            memset(instance_coefficients, 0, instance->dict->word_count * sizeof(f32));
            memcpy(instance_coefficients, coefs, count * sizeof(f32));
            instance->needs_recompute = true;
        }
    }
}

internal void set_preset(Instance *instance, i32 coef_set, i32 preset)
{
    if (preset > 0) {
        if (coef_set == CoefSet_C || coef_set == CoefSet_D) {
            // A little janky: sending 0 to either c or d makes the interp linear instead of bilinear
            // But is ignored if sent to a or b
            instance->bilinear = true;
        }

        set_all(instance, coef_set, instance->dict->word_count, get_preset(instance->dict, preset - 1));
    } else {
        if (coef_set == CoefSet_C || coef_set == CoefSet_D) {
            instance->bilinear = false;
            instance->needs_recompute = true; // Too easy to forget about
        }
    }
}

internal f32 get_sample_buffers(Instance *instance, f32 t, f32 **left, f32 **right)
{
    f32 t_scale = t * (InterpolationNodes - 1);
    isize t_i = (isize)t_scale;
    f32 t_f = t_scale - t_i;

    *left = instance->sample_bufs[0] + (t_i * instance->dict->sample_len);
    *right = instance->sample_bufs[1] + (t_i * instance->dict->sample_len);

    return t_f;
}

typedef struct sample_buffers_bilinear
{
    f32 x, y;
    f32 *left_a, *right_a;
    f32 *left_b, *right_b;
    f32 *left_c, *right_c;
    f32 *left_d, *right_d;
} sample_buffers_bilinear;

internal sample_buffers_bilinear get_sample_buffers_bilinear(Instance *instance, f32 t_x, f32 t_y)
{
    f32 t_x_scale = clamp01(t_x) * (InterpolationNodesBilinear - 1);
    f32 t_y_scale = clamp01(t_y) * (InterpolationNodesBilinear - 1);
    isize t_x_i = (isize)t_x_scale;
    f32 t_x_f = t_x_scale - t_x_i;
    isize t_y_i = (isize)t_y_scale;
    f32 t_y_f = t_y_scale - t_y_i;

    isize x_stride = instance->dict->sample_len;
    isize y_stride = instance->dict->sample_len * InterpolationNodesBilinear;
    isize x_offset = t_x_i * x_stride;
    isize y_offset = t_y_i * y_stride;

    f32 *left_a = instance->sample_bufs[0] + x_offset + y_offset;
    f32 *right_a = instance->sample_bufs[1] + x_offset + y_offset;
    assert(left_a < instance->sample_bufs[1]);
    assert(right_a < instance->sample_end);

    sample_buffers_bilinear result = (sample_buffers_bilinear) {
        .x = t_x_f,
        .y = t_y_f,
        .left_a = left_a,
        .left_b = left_a + x_stride,
        .left_c = left_a + y_stride,
        .left_d = left_a + y_stride + x_stride,
        .right_a = right_a,
        .right_b = right_a + x_stride,
        .right_c = right_a + y_stride,
        .right_d = right_a + y_stride + x_stride
    };

    if (t_x == 1.f) {
        result.left_b = result.left_a;
        result.left_d = result.left_c;
        result.right_b = result.right_a;
        result.right_d = result.right_c;
    }

    if (t_y == 1.f) {
        result.left_c = result.left_a;
        result.left_d = result.left_b;
        result.right_c = result.right_a;
        result.right_d = result.right_b;
    }

    return result;
}

internal void set_t_x(Instance *instance, f32 t)
{
    instance->t_x = smooth(clamp01(t));
}

internal void set_t_y(Instance *instance, f32 t)
{
    instance->t_y = smooth(clamp01(t));
}

internal void set_dictionary(Instance *instance, char *dict_name)
{
    Dict *dict = get_dict(dict_name);

    if (dict) {
        instance->dict = dict;
        instance->needs_recompute = true;
    }
}

internal void recompute(Instance *instance)
{
    if (instance->needs_recompute) {
        interpolate(instance);
    }
}

internal isize new(char const *dict_name, u8 *buf, isize buf_len)
{
    assert(IS_ALIGNED_64((size_t)buf));

    Stack *stack = &(Stack) {
        buf, buf, buf + buf_len
    };

    Dict *dict = get_dict(dict_name);
    if (dict == 0) {
        dict = &empty_dict;
    }

    isize sample_bufs = max(InterpolationNodes, InterpolationNodesBilinear * InterpolationNodesBilinear);

    Instance *i = stack_push_one(stack, Instance);

    i->dict = dict;
    i->a_coefficients = stack_push_n(stack, f32, dict->word_count);
    i->b_coefficients = stack_push_n(stack, f32, dict->word_count);
    i->c_coefficients = stack_push_n(stack, f32, dict->word_count);
    i->d_coefficients = stack_push_n(stack, f32, dict->word_count);
    i->scratch[0] = stack_push_n(stack, f32, dict->word_len);
    i->scratch[1] = stack_push_n(stack, f32, dict->word_len);
    i->sample_bufs[0] = stack_push_n(stack, f32, sample_bufs * (dict->sample_len));
    i->sample_bufs[1] = stack_push_n(stack, f32, sample_bufs * (dict->sample_len));
    i->sample_end = stack_push_n(stack, f32, 1);

    return stack->ptr - stack->buf;
}

internal isize new_size_req(char const *dict_name)
{
    return new(dict_name, DRY_RUN, 0);
}

isize first_nonzero(f64 *buf, isize len)
{
    for (isize i = 0; i < len; i++) {
        if (buf[i]) {
            return i;
        }
    }
    return -1;
}

internal void zero_buffers_stereo(f64 **outs, isize samples)
{
    memset(outs[0], 0, samples * sizeof(f64));
    memset(outs[1], 0, samples * sizeof(f64));
}

internal void max_perform(void *x, void *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, isize instance_offset)
{
    // This is gauranteed by the sparse-interp~ host object.
    assert(numouts == 2);

    Instance *instance = *(Instance **)((u8 *)x + instance_offset);

    if (instance == 0) {
        zero_buffers_stereo(outs, sampleframes);
        return;
    }

    Dict *dict = instance->dict;

    if (dict == 0) {
        zero_buffers_stereo(outs, sampleframes);
        return;
    }

    u32 ftz =_MM_GET_FLUSH_ZERO_MODE();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    (void)x, dsp64, ins, numins, flags, numouts;
    assert(IS_POWER_2(sampleframes));

    usize wrap_mask = dict->sample_len - 1;
    u32 s = instance->sample_index;
    f64 *envelope = ins[0];
    f64 *left = outs[0];
    f64 *right = outs[1];
    f32 t_x = instance->t_x;
    f32 t_x_z = instance->t_x_z;
    f32 t_y = instance->t_y;
    f32 t_y_z = instance->t_y_z;

#if 0
    isize start = first_nonzero(envelope, sampleframes);

    if (start == -1) {
        memset(left, 0, sampleframes * sizeof(f64));
        memset(right, 0, sampleframes * sizeof(f64));
        start = sampleframes;
        t_z = t;
    } else if (start != 0) {
        assert(start > 0);
        memset(left, (i32)start, sampleframes * sizeof(f64));
        memset(right, (i32)start, sampleframes * sizeof(f64));
        s = 0;
    }
#else
    isize start = 0;

    // The else case above seems to never happen. So this suffices:
    if (envelope[0] == 0) {
        zero_buffers_stereo(outs, sampleframes);
        instance->t_x_z = t_x;
        instance->t_y_z = t_y;
        instance->active = false;
        return;
    }
#endif

    instance->active = true;

    isize next_sample_offset = dict->sample_len;

    if (instance->bilinear) {
        for (isize i = start; i < sampleframes; i++) {
            t_x_z = t_x * T_B + t_x_z * T_A;
            t_y_z = t_y * T_B + t_y_z * T_A;

            sample_buffers_bilinear sbb = get_sample_buffers_bilinear(instance, t_x_z, t_y_z);

            f64 e = envelope[i];
            usize idx = s & wrap_mask;

            f32 lab = lerp(sbb.left_a[idx], sbb.left_b[idx], sbb.x);
            f32 lcd = lerp(sbb.left_c[idx], sbb.left_d[idx], sbb.x);
            f32 labcd = lerp(lab, lcd, sbb.y);

            f32 rab = lerp(sbb.right_a[idx], sbb.right_b[idx], sbb.x);
            f32 rcd = lerp(sbb.right_c[idx], sbb.right_d[idx], sbb.x);
            f32 rabcd = lerp(rab, rcd, sbb.y);

            left[i] = e * labcd;
            right[i] = e * rabcd;
            s++;
        }
    } else {
        for (isize i = start; i < sampleframes; i++) {
            t_x_z = t_x * T_B + t_x_z * T_A;

            f32 *left_sample, *right_sample;
            f32 t_f = get_sample_buffers(instance, t_x_z, &left_sample, &right_sample);

            f64 e = envelope[i];

            usize idx = s & wrap_mask;
            left[i] = e * lerp(left_sample[idx], (left_sample + next_sample_offset)[idx], t_f);
            right[i] = e * lerp(right_sample[idx], (right_sample + next_sample_offset)[idx], t_f);
            s++;
        }
    }

    instance->sample_index = s;
    instance->t_x_z = t_x_z;
    instance->t_y_z = t_y_z;
     _MM_SET_FLUSH_ZERO_MODE(ftz);
}

__declspec(dllexport) void init_api(struct sparse_interp_api *api)
{
    api->init_api = init_api;
    api->init_size_req = init_size_req;
    api->init = init;
    api->new_size_req = new_size_req;
    api->new = new;
    api->load_dictionary = load_dictionary;
    api->load_dictionary_size_req = load_dictionary_size_req;
    api->note = note;
    api->set_coefficients = set_coefficients;
    api->set_all = set_all;
    api->set_preset = set_preset;
    api->set_t_x = set_t_x;
    api->set_t_y = set_t_y;
    api->recompute = recompute;
    api->max_perform = (void*)max_perform;
}

#if STANDALONE == 1

internal void *alloc64(isize size) {
    u8 *allocation = malloc(size + 64);
    assert(allocation);
    u8 *aligned = allocation;
    if (IS_ALIGNED_64(allocation)) {
        aligned += 8;
    }

    aligned = (u8 *)ROUND_UP_64((size_t)aligned);
    assert(aligned - allocation >= 8);
    u32 *free_cookie = (u32 *)aligned - 1;
    u32 *free_mark = (u32 *)aligned - 2;
    *free_cookie = (u32)HASH("alloc64");
    *free_mark = (u32)(aligned - allocation);

    assert(aligned - *free_mark == allocation);
    return aligned;
}

internal void free_alloc(void *p)
{
    u32 *cookie = (u32 *)p - 1;
    if (*cookie == (u32)HASH("alloc64")) {
        u32 *offset = (u32 *)p - 2;
        assert (*offset <= 64);
        u8 *alloc = (u8 *)p - *offset;
        free(alloc);
    } else {
        assert(!"free_alloc: alloc64 cookie missing");
    }
}

static char const *test_dict_name = ".\\..\\Dump.npy";
static char const *test_presets_name = ".\\..\\Dump coefficients.npy";

internal void test_init()
{
    State *s = 0;

    {
        isize size = init_size_req(0);
        s = alloc64(size);
        isize ret = init(0, (u8 *)s, size);
        assert(ret == size);
    }

    {
        isize size = init_size_req(s);
        assert(size == 0);
        isize ret = init(s, 0, 0);
        assert(ret == size);
    }

    {
        isize dict_size = load_dictionary_size_req(test_dict_name, test_presets_name);
        assert(dict_size);
        u8 *buf = alloc64(dict_size);
        load_dictionary(test_dict_name, test_presets_name, buf, dict_size);
    }

    {
        isize size = new_size_req(0);
        assert(size > 0);
        u8 *buf = alloc64(size);
        isize ret = new(0, buf, size);
        assert(ret == size);
        free_alloc(buf);
    }

    {
        isize size = new_size_req(test_dict_name);
        assert(size > 0);
        u8 *buf = alloc64(size);
        isize ret = new(test_dict_name, buf, size);
        assert(ret == size);
        free_alloc(buf);
    }

    Dict *dict = get_dict(test_dict_name);
    assert(dict);
    printf("buf_dim { %d, %d }\nword_len %d\nword_count %d\npreset_count %d\n", dict->buf_dim.x, dict->buf_dim.y, dict->word_len, dict->word_count, dict->preset_count);
}

internal void test_interpolate()
{
    // Must be run after test_init. Bad test hygiene. Whatever.

    isize size = new_size_req(test_dict_name);
    Instance *instance = alloc64(size);
    new(test_dict_name, (u8 *)instance, size);

    set_preset(instance, CoefSet_A, 1);
    set_preset(instance, CoefSet_B, 2);
    set_preset(instance, CoefSet_C, 0);
    set_t_x(instance, 0);
    note(instance, 1);

    long frames = 64;
    f64 *ins[1] = { alloc64(frames * sizeof(f64)) };
    f64 *outs[2] = { ins[0], alloc64(frames * sizeof(f64))};
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1e-6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 0.5);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1 - 1e6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);

    for (isize i = 0; i < frames; i++) {
        ins[0][i] = (f64)i / (f64)frames;
    }
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1e-6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 0.5);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1 - 1e6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);

    set_preset(instance, CoefSet_C, 3);
    set_preset(instance, CoefSet_D, 4);

    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1e-6);
    set_t_y(instance, 1e-6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 0.5);
    set_t_y(instance, 0.5);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1 - 1e6);
    set_t_y(instance, 1 - 1e6);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);
    set_t_x(instance, 1);
    set_t_y(instance, 1);
    max_perform(&instance, 0, ins, 1, outs, 2, frames, 0, 0);

    free_alloc(instance);
    free_alloc(ins[0]);
    free_alloc(outs[1]);
}

internal void test_hashtable()
{
    Dict *a = get_dict("blah");
    assert(a == 0);
    Dict *b = &(Dict){ "b" };
    add_dict(b);
    Dict *bb = get_dict(b->name);
    assert(bb == b);
}

i32 main(i32 argc, char **argv)
{
    (void)argc, argv;
    test_init();
    test_interpolate();
    test_hashtable();

    return 0;
}

#endif // STANDALONE
#endif // MAX_EXTERNAL