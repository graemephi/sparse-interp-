/**
    sparse-interp~
*/

#ifdef __APPLE__
#include "Carbon/Carbon.h"
#endif

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"

#ifdef _MSC_VER
#include "common/dllmain_win.c"
#endif

#ifndef DYNAMIC_LOADING
#define DYNAMIC_LOADING 0
#endif

#ifndef MAX_EXTERNAL
#define MAX_EXTERNAL 1
#endif

#include "dsp.c"

#define verbose(...)
// #define verbose(x, str, ...) object_post((t_object *)x, str, __VA_ARGS__)

enum {
    MaxCoefficients = 128,

    Inlet_Note = 0,
    Inlet_A,
    Inlet_B,
    Inlet_C,
    Inlet_D,
    Inlet_T_X,
    Inlet_T_Y
};

#if DYNAMIC_LOADING == 1
static const char *ext_watch_path = "C:\\code\\sparse-interp\\mxe\\build\\dsp.dll";
static const char *ext_load_path = "C:\\code\\sparse-interp\\mxe\\build\\dsp.fw.dll";
static const char *ext_load_path_alt = "C:\\code\\sparse-interp\\mxe\\build\\dsp.fw2.dll";
#endif

typedef struct sparse_interp {
    t_pxobject hdr;

    void *instance;
    isize instance_size;

    long active_inlet;
    void *proxy[6];

    char const *dict;

    // Buffers for set function arguments--doesn't reflect instance state
    int *indices;
    float *values;

    // Hack: We use these to allow swapping dictionaries. But it only works with presets
    int a, b, c, d;
    float t_x, t_y;
} sparse_interp;

typedef struct filewatch {
    t_object hdr;
    void *filewatcher_handle;
} filewatch;

void void_noop() {

}

void *void_star_noop() {
    return 0;
}

int int_noop() {
    return 0;
}

size_t size_t_noop() {
    return 64;
}

// We make very few allocations, so the uneccesarily extreme alignment
// requirement is fine.
void *alloc64(isize size) {
    u8 *allocation = sysmem_newptr((long)size + 64);
    u8 *aligned = allocation;
    assert(allocation);
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

void free_alloc(void *p)
{
    u32 *cookie = (u32 *)p - 1;
    if (*cookie == (u32)HASH("alloc64")) {
        u32 *offset = (u32 *)p - 2;
        assert (*offset <= 64);
        u8 *alloc = (u8 *)p - *offset;
        sysmem_freeptr(alloc);
    } else {
        assert(!"free_alloc: alloc64 cookie missing");
    }
}


void void_perform(void *x, void *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userdata)
{

}

static sparse_interp_api si = {0};

static t_class *si_class = 0;

static t_object *dsp_chain = 0;

static void *sparse_interp_state = 0;

#define dict_ext ".npy"
#define dict_presets " coefficients" dict_ext

char * to_absolute_path(char *path)
{
    // Modifies path.

    char temp[MAX_FILENAME_CHARS];

    short path_id;
    t_fourcc cc;
    short ret = locatefile_extended(path, &path_id, &cc, 0, 0);

    if (ret == 0) {
        path_toabsolutesystempath(path_id, path, temp);

#ifdef _MSC_VER
        path_nameconform(temp, path, PATH_STYLE_NATIVE, PATH_TYPE_PATH);
#endif

        return gensym(temp)->s_name;
    }

    return 0;
}

void make_instance(sparse_interp *x, char const *dict)
{
    // Todo: dont do this every time
    char dict_path[MAX_FILENAME_CHARS];
    char presets_path[MAX_FILENAME_CHARS];
    snprintf(dict_path, sizeof(dict_path), "%s%s", dict, dict_ext);
    snprintf(presets_path, sizeof(presets_path), "%s%s", dict, dict_presets);

    char *abs_dict = to_absolute_path(dict_path);
    char *abs_presets = to_absolute_path(presets_path);

    if (abs_dict == 0 || abs_presets ==0) {
        object_post((t_object *)x, "Failed to load dictionary %s", dict);
        return;
    }

    isize dict_size = si.load_dictionary_size_req(abs_dict, abs_presets);

    if (dict_size) {
        u8 *buf = alloc64(dict_size);
        si.load_dictionary(abs_dict, abs_presets, buf, dict_size);
    }

    isize instance_size = si.new_size_req(abs_dict);
    assert(instance_size > 0);

    if (x->instance && instance_size > x->instance_size) {
        free_alloc(x->instance);
        x->instance = 0;
    }

    if (x->instance == 0) {
        x->instance = alloc64(instance_size);
        x->instance_size = instance_size;
    }

    isize used = si.new(abs_dict, x->instance, instance_size);
    if (used != instance_size) {
        object_error((t_object *)x, "sparse-interp~ failed initialisation");
        free_alloc(x->instance);
        x->instance = 0;
        x->instance_size = 0;
    } else {
        x->dict = dict;

        si.set_preset(x->instance, CoefSet_A, x->a);
        si.set_preset(x->instance, CoefSet_B, x->b);
        si.set_preset(x->instance, CoefSet_C, x->c);
        si.set_preset(x->instance, CoefSet_D, x->d);
        si.set_t_x(x->instance, x->t_x);
        si.set_t_y(x->instance, x->t_y);
    }
}

i32 inlet_to_coef_set(i32 inlet)
{
    i32 result = 0;
    switch (inlet) {
        case Inlet_A: { result = CoefSet_A; } break;
        case Inlet_B: { result = CoefSet_B; } break;
        case Inlet_C: { result = CoefSet_C; } break;
        case Inlet_D: { result = CoefSet_D; } break;
        default: assert(0);
    }
    return result;
}

char inlet_to_coef_char(i32 inlet)
{
    char result = 0;
    switch (inlet) {
        case Inlet_A: { result = 'a'; } break;
        case Inlet_B: { result = 'b'; } break;
        case Inlet_C: { result = 'c'; } break;
        case Inlet_D: { result = 'd'; } break;
        default: assert(0);
    }
    return result;
}

void *si_new(t_symbol *s, long argc, t_atom *argv)
{
    sparse_interp *x = (sparse_interp *)object_alloc(si_class);
    assert(x->instance_size == 0);

    if (x) {
        x->indices = alloc64(sizeof(int) * MaxCoefficients);
        x->values = alloc64(sizeof(float) * MaxCoefficients);

        const char *dict = 0;
        if (argc > 0 && (atom_gettype(argv) == A_SYM)) {
            dict = atom_getsym(argv)->s_name;
            make_instance(x, dict);
        }

        dsp_setup(&x->hdr, 1);
        outlet_new(x, "signal");
        outlet_new(x, "signal");
        x->proxy[5] = proxy_new((t_object *)x, 6, &x->active_inlet);
        x->proxy[4] = proxy_new((t_object *)x, 5, &x->active_inlet);
        x->proxy[3] = proxy_new((t_object *)x, 4, &x->active_inlet);
        x->proxy[2] = proxy_new((t_object *)x, 3, &x->active_inlet);
        x->proxy[1] = proxy_new((t_object *)x, 2, &x->active_inlet);
        x->proxy[0] = proxy_new((t_object *)x, 1, &x->active_inlet);

        verbose((t_object *)x, "sparse-interp~ initialised");
    }

    return x;
}

void si_free(sparse_interp *x)
{
    if (x->instance) {
        free_alloc(x->instance);
    }

    free_alloc(x->indices);
    free_alloc(x->values);

    dsp_free((t_pxobject *)x);

    object_free(x->proxy[0]);
    object_free(x->proxy[1]);
    object_free(x->proxy[2]);
}


void si_assist(sparse_interp *x, void *b, long m, long a, char *s)
{
    if (m == ASSIST_INLET) {
        switch (a) {
            case Inlet_Note: {
                sprintf(s, "numeric: Note on velocity");
            } break;
            case Inlet_A: {
                sprintf(s, "a. int: preset, int float list: index coef ..., float list: coefficients");
            } break;
            case Inlet_B: {
                sprintf(s, "b. int: preset, int float list: index coef ..., float list: coefficients");
            } break;
            case Inlet_C: {
                sprintf(s, "c. int: preset, int float list: index coef ..., float list: coefficients");
            } break;
            case Inlet_D: {
                sprintf(s, "d. int: preset, int float list: index coef ..., float list: coefficients");
            } break;
            case Inlet_T_X: {
                sprintf(s, "t x. numeric: interpolation");
            } break;
            case Inlet_T_Y: {
                sprintf(s, "t y. numeric: interpolation");
            } break;
            default: {
                assert(0);
            }
        }
    } else {
        if (a == 0) {
            sprintf(s, "signal: left");
        } else if (a == 1) {
            sprintf(s, "signal: right");
        }
    }
}

void si_list(sparse_interp *x, t_symbol *msg, int argc, t_atom *argv) {
    t_atom *first = argv;
    long first_type = atom_gettype(first);

    long inlet = proxy_getinlet((t_object *)x);

    if (x->instance == 0) {
        return;
    }

    if (inlet != Inlet_A || inlet != Inlet_B || inlet != Inlet_C || inlet != Inlet_D) {
        return;
    }

    memset(x->indices, 0, MaxCoefficients * sizeof(f32));
    memset(x->values, 0, MaxCoefficients * sizeof(f32));

    if (first_type == A_LONG) {
        // Sequence of index value pairs

        if ((argc & 1) == 1) {
            object_error((t_object *)x, "%c inlet: pair list is uneven.", inlet == Inlet_A ? 'a' : 'b');
            return;
        }

        int len = argc / 2;

        if (len > MaxCoefficients) {
            object_error((t_object *)x, "%c inlet: pair list is too long (got %d, max %d)", inlet == Inlet_A ? 'a' : 'b', len, MaxCoefficients);
            return;
        }

        t_atom *arg_cursor = argv;
        for (int i = 0; i < len; i++) {
            t_atom *index_atom = arg_cursor;
            t_atom *value_atom = arg_cursor + 1;

            int index = 0;
            float value = 0;

            if (atom_gettype(index_atom) == A_LONG) {
                index = (int)atom_getlong(index_atom);
            } else {
                object_error((t_object *)x, "%c inlet: non-integer index", inlet == Inlet_A ? 'a' : 'b');
                return;
            }

            if (atom_gettype(value_atom) == A_FLOAT) {
                value = (float)atom_getfloat(value_atom);
            } else if (atom_gettype(value_atom) == A_LONG) {
                value = (float)atom_getlong(value_atom);
            } else {
                object_error((t_object *)x, "%c inlet: non-numeric value", inlet == Inlet_A ? 'a' : 'b');
                return;
            }

            x->indices[i] = index;
            x->values[i] = value;

            arg_cursor += 2;
        }

        verbose(x, "si.set_coefficients(%c, ... %d ...)", inlet_to_coef_char(inlet), len);
        si.set_coefficients(x->instance, inlet_to_coef_set(inlet), len, x->indices, x->values);
    } else if (first_type == A_FLOAT) {
        // Sequence of coefficients in order

        int len = argc;

        if (len > MaxCoefficients) {
            object_error((t_object *)x, "%c inlet: list is too long (got %d, max %d)", inlet == Inlet_A ? 'a' : 'b', len, MaxCoefficients);
            return;
        }

        t_atom *arg_cursor = argv;
        for (int i = 0; i < len; i++) {
            t_atom *value_atom = arg_cursor;

            float value = 0;

            if (atom_gettype(value_atom) == A_FLOAT) {
                value = (float)atom_getfloat(value_atom);
            } else if (atom_gettype(value_atom) == A_LONG) {
                value = (float)atom_getlong(value_atom);
            } else {
                object_error((t_object *)x, "%c inlet: non-numeric value", inlet == Inlet_A ? 'a' : 'b');
                return;
            }

            x->values[i] = value;

            arg_cursor++;
        }


        verbose(x, "si.set_all(%c, ... %d ...)", inlet_to_coef_char(inlet), len);
        si.set_all(x->instance, inlet_to_coef_set(inlet), len, x->values);
    }
}

void si_int(sparse_interp *x, int i)
{
    if (x->instance == 0) {
        return;
    }

    long inlet = proxy_getinlet((t_object *)x);

    switch (inlet) {
        case Inlet_Note: {
            verbose(x, "si.note(%d)", i);
            si.note(x->instance, (float)i);
        } break;
        case Inlet_A: {
            verbose(x, "si.set_preset(%c, %d)", inlet_to_coef_char(inlet), i);
            x->a = i;
            si.set_preset(x->instance, inlet_to_coef_set(inlet), i);
        } break;
        case Inlet_B: {
            verbose(x, "si.set_preset(%c, %d)", inlet_to_coef_char(inlet), i);
            x->b = i;
            si.set_preset(x->instance, inlet_to_coef_set(inlet), i);
        } break;
        case Inlet_C: {
            verbose(x, "si.set_preset(%c, %d)", inlet_to_coef_char(inlet), i);
            x->c = i;
            si.set_preset(x->instance, inlet_to_coef_set(inlet), i);
        } break;
        case Inlet_D: {
            verbose(x, "si.set_preset(%c, %d)", inlet_to_coef_char(inlet), i);
            x->d = i;
            si.set_preset(x->instance, inlet_to_coef_set(inlet), i);
        } break;
        case Inlet_T_X: {
            verbose(x, "si.set_t_x(%d)", i);
            x->t_x = (float)i;
            si.set_t_x(x->instance, (float)i);
        } break;
        case Inlet_T_Y: {
            verbose(x, "si.set_t_y(%d)", i);
            x->t_y = (float)i;
            si.set_t_y(x->instance, (float)i);
        } break;
        default: {
            assert(0);
        }
    }
}

void si_float(sparse_interp *x, double f)
{
    if (x->instance == 0) {
        return;
    }

    long inlet = proxy_getinlet((t_object *)x);

    if (inlet == 0) {
        verbose(x, "si.note(%.3f)", (float)f);
        si.note(x->instance, (float)f);
    } else if (inlet == Inlet_T_X) {
        verbose(x, "si.set_t_x(%.3f)", f);
        x->t_x = (float)f;
        si.set_t_x(x->instance, (float)f);
    } else if (inlet == Inlet_T_Y) {
        verbose(x, "si.set_t_y(%.3f)", f);
        x->t_y = (float)f;
        si.set_t_y(x->instance, (float)f);
    }
}

void si_set(sparse_interp *x, t_symbol *sym)
{
    char *dict = sym->s_name;
    if (x->dict != dict) {
        verbose(x, "Setting dictionary to %s", dict);
        make_instance(x, dict);
    }
}

void si_bang(sparse_interp *x)
{
    if (x->instance) {
        si.recompute(x->instance);
    }
}

void si_dsp64(sparse_interp *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
    dsp_chain = dsp64;

    verbose(x, "sample rate is %f. perform is %s", samplerate, (si.max_perform == void_perform) ? "no-op" : "available");
    if (count[0]) {
        object_method(dsp64, gensym("dsp_add64"), x, si.max_perform, 0, offsetof(sparse_interp, instance));
    }
}

void stub_library()
{
    // todo: even check calling conventions make this ok
    si.init_api = (void *)void_noop;
    si.init_size_req = (void *)size_t_noop;
    si.init = (void *)void_star_noop;
    si.new_size_req = (void *)size_t_noop;
    si.new = (void *)int_noop;
    si.note = (void *)void_noop;
    si.set_coefficients = (void *)void_noop;
    si.set_all = (void *)void_noop;
    si.set_preset = (void *)void_noop;
    si.set_t_x = (void *)void_noop;
    si.set_t_y = (void *)void_noop;
    si.set_dictionary = (void *)void_noop;
    si.recompute = (void *)void_noop;
    si.max_perform = (void *)void_perform;
}

#if DYNAMIC_LOADING == 1
static t_class *fw_class = 0;
static filewatch *fw = 0;
#endif

void load_library()
{
#if DYNAMIC_LOADING == 1
    // note: In general this whole mechanism is currently very un-thread safe.

    stub_library();

    if (si.handle) {
        FreeLibrary(si.handle);
        si.handle = 0;
    }

    if (CopyFile(ext_watch_path, ext_load_path, false) == 0) {
        // ext_load_path is probably locked by another process.
        ext_load_path = ext_load_path_alt;
        if (CopyFile(ext_watch_path, ext_load_path, false) == 0) {
            object_error(&fw->hdr, "Could not copy external. (Windows error code: %d)", GetLastError());
            return;
        }
	}

	si.handle = LoadLibrary(ext_load_path);
	assert(si.handle);
	void (*init_api)(sparse_interp_api *api) = (void *)GetProcAddress(si.handle, "init_api");
#endif

	init_api(&si);
    assert(si.init_api == init_api);

    isize buf_len = si.init_size_req(sparse_interp_state);
    void *buf = 0;
    if (buf_len) {
        buf = alloc64(buf_len);
        assert(buf);
    }

    isize used = si.init(sparse_interp_state, buf, buf_len);
    assert(used == buf_len);

    if (buf_len == 0) {
        free_alloc(sparse_interp_state);
        sparse_interp_state = buf;
    }
}

#if DYNAMIC_LOADING == 1
void *fw_new(t_symbol *s, long argc, t_atom *argv)
{
    assert(fw == 0);
    filewatch *x = object_alloc(fw_class);

    if (x) {
        short path_id;
        t_fourcc cc;

		char filename[MAX_FILENAME_CHARS];
		strncpy_zero(filename, ext_watch_path, MAX_FILENAME_CHARS);
		short ret = locatefile_extended(filename, &path_id, &cc, 0, 0);
		assert(ret == 0);

		x->filewatcher_handle = filewatcher_new((t_object *)x, path_id, filename);
		assert(x->filewatcher_handle);

		filewatcher_start(x->filewatcher_handle);
    }

    return x;
}

void fw_filechanged(filewatch *x, char *filename, short path)
{
    int dsp_state = sys_getdspstate();

    // Probably we need to do more here and ensure the audio thread has actually stopped processing audio
    // Otherwise it might try to call into the dsp procedure we're about to unload.
    // CRASH COUNT: 2. Was it this? Who can say!
    // It was this :(

    if (dsp_state) {
        canvas_stop_dsp();
    }

    if (dsp_chain) {
        dspchain_setbroken((t_dspchain *)dsp_chain);
        dsp_chain = 0;
    }

    load_library();

    if (dsp_state) {
        canvas_start_dsp();
    }
}

void fw_free(filewatch *x)
{
    object_free(x->filewatcher_handle);

    if (si.handle) {
        FreeLibrary(si.handle);
        si.handle = 0;
    }
}
#endif

void ext_main(void *r)
{
    load_library();

    si_class = class_new("sparse-interp~", (method)si_new, (method)si_free, (long)sizeof(sparse_interp), 0L, A_GIMME, 0);
    class_addmethod(si_class, (method)si_float, "float", A_FLOAT, 0);
    class_addmethod(si_class, (method)si_dsp64, "dsp64", A_CANT, 0);
    class_addmethod(si_class, (method)si_bang, "bang", 0);
    class_addmethod(si_class, (method)si_int, "int", A_LONG, 0);
    class_addmethod(si_class, (method)si_list, "list", A_GIMME, 0);
    class_addmethod(si_class, (method)si_set, "set", A_SYM, 0);
    class_addmethod(si_class, (method)si_assist, "assist", A_CANT, 0);
    class_dspinit(si_class);
    class_register(CLASS_BOX, si_class);

#if DYNAMIC_LOADING == 1
    fw_class = class_new("sparse-interp-filewatch", (method)fw_new, (method)fw_free, (long)sizeof(filewatch), 0L, A_GIMME, 0);
    class_addmethod(fw_class, (method)fw_filechanged, "filechanged", A_CANT, 0);
    class_register(CLASS_NOBOX, fw_class);
    fw = object_new(CLASS_NOBOX, gensym("sparse-interp-filewatch"));
#endif
}
