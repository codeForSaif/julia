// This file is a part of Julia. License is MIT: http://julialang.org/license

//===----------------------------------------------------------------------===//
//
// The ABI implementation used for AArch64 targets.
//
//===----------------------------------------------------------------------===//
//
// The Procedure Call Standard can be found here:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
//
//===----------------------------------------------------------------------===//

namespace {

typedef bool AbiState;
static const AbiState default_abi_state = 0;

static Type *get_llvm_fptype(jl_datatype_t *dt)
{
    // Assume jl_is_datatype(dt) && !jl_is_abstracttype(dt)
    if (dt->mutabl || jl_datatype_nfields(dt) != 0)
        return NULL;
    Type *lltype;
    // Check size first since it's cheaper.
    switch (dt->size) {
    case 2:
        lltype = T_float16;
        break;
    case 4:
        lltype = T_float32;
        break;
    case 8:
        lltype = T_float64;
        break;
    case 16:
        lltype = T_float128;
        break;
    default:
        return NULL;
    }
    return jl_is_floattype((jl_value_t*)dt) ? lltype : NULL;
}

// Whether a type is a homogeneous floating-point aggregates (HFA) or a
// homogeneous short-vector aggregates (HVA). Returns the number of members.
// We only handle HFA of HP, SP and DP here since these are the only ones we
// have (no QP).
static size_t isHFAorHVA(jl_datatype_t *dt)
{
    // Assume jl_is_datatype(dt) && !jl_is_abstracttype(dt)

    // An Homogeneous Floating-point Aggregate (HFA) is an Homogeneous Aggregate
    // with a Fundamental Data Type that is a Floating-Point type and at most
    // four uniquely addressable members.
    // An Homogeneous Short-Vector Aggregate (HVA) is an Homogeneous Aggregate
    // with a Fundamental Data Type that is a Short-Vector type and at most four
    // uniquely addressable members.
    size_t members = jl_datatype_nfields(dt);
    if (members < 1 || members > 4)
        return 0;
    // There's at least one member
    jl_value_t *ftype = jl_field_type(dt, 0);
    if (!get_llvm_fptype((jl_datatype_t*)ftype))
        return 0;
    for (size_t i = 1;i < members;i++) {
        if (ftype != jl_field_type(dt, i)) {
            return 0;
        }
    }
    return members;
}

void needPassByRef(AbiState*, jl_value_t *ty, bool *byRef, bool*)
{
    // Assume jl_is_datatype(ty) && !jl_is_abstracttype(ty)
    jl_datatype_t *dt = (jl_datatype_t*)ty;
    // B.2
    //   If the argument type is an HFA or an HVA, then the argument is used
    //   unmodified.
    if (isHFAorHVA(dt))
        return;
    // B.3
    //   If the argument type is a Composite Type that is larger than 16 bytes,
    //   then the argument is copied to memory allocated by the caller and the
    //   argument is replaced by a pointer to the copy.
    // We only check for the total size and not whether it is a composite type
    // since there's no corresponding C type and we just treat such large
    // bitstype as a composite type of the right size.
    *byRef = dt->size > 16;
    // B.4
    //   If the argument type is a Composite Type then the size of the argument
    //   is rounded up to the nearest multiple of 8 bytes.
}

bool need_private_copy(jl_value_t*, bool)
{
    return false;
}

// Determine which kind of register the argument will be passed in and
// if the argument has to be passed on stack (including by reference).
//
// If the argument should be passed in SIMD and floating-point registers,
// we may need to rewrite the argument types to [n x ftype].
// If the argument should be passed in general purpose registers, we may need
// to rewrite the argument types to [n x i64].
//
// If the argument has to be passed on stack, we need to use sret.
//
// All the out parameters should be default to `false`.
static void classify_arg(jl_value_t *ty, bool *fpreg, bool *onstack,
                         bool *need_rewrite)
{
    // Assume jl_is_datatype(ty) && !jl_is_abstracttype(ty)
    jl_datatype_t *dt = (jl_datatype_t*)ty;

    // Based on section 5.4 C of the Procedure Call Standard
    // C.1
    //   If the argument is a Half-, Single-, Double- or Quad- precision
    //   Floating-point or Short Vector Type and the NSRN is less than 8, then
    //   the argument is allocated to the least significant bits of register
    //   v[NSRN]. The NSRN is incremented by one. The argument has now been
    //   allocated.
    // Note that this is missing QP float as well as short vector types since we
    // don't really have those types.
    if (get_llvm_fptype(dt)) {
        *fpreg = true;
        return;
    }

    // C.2
    //   If the argument is an HFA or an HVA and there are sufficient
    //   unallocated SIMD and Floating-point registers (NSRN + number of
    //   members <= 8), then the argument is allocated to SIMD and
    //   Floating-point Registers (with one register per member of the HFA
    //   or HVA). The NSRN is incremented by the number of registers used.
    //   The argument has now been allocated.
    if (isHFAorHVA(dt)) { // HFA and HVA have <= 4 members
        *fpreg = true;
        *need_rewrite = true;
        return;
    }

    // Check if the argument needs to be passed by reference. This should be
    // done before starting step C but we do this here to avoid checking for
    // HFA and HVA twice. We don't check whether it is a composite type.
    // See `needPassByRef` above.
    if (dt->size > 16) {
        *onstack = true;
        return;
    }

    // C.3
    //   If the argument is an HFA or an HVA then the NSRN is set to 8 and the
    //   size of the argument is rounded up to the nearest multiple of 8 bytes.
    // C.4
    //   If the argument is an HFA, an HVA, a Quad-precision Floating-point or
    //   Short Vector Type then the NSAA is rounded up to the larger of 8 or
    //   the Natural Alignment of the argument’s type.
    // C.5
    //   If the argument is a Half- or Single- precision Floating Point type,
    //   then the size of the argument is set to 8 bytes. The effect is as if
    //   the argument had been copied to the least significant bits of a 64-bit
    //   register and the remaining bits filled with unspecified values.
    // C.6
    //   If the argument is an HFA, an HVA, a Half-, Single-, Double- or
    //   Quad- precision Floating-point or Short Vector Type, then the argument
    //   is copied to memory at the adjusted NSAA. The NSAA is incremented
    //   by the size of the argument. The argument has now been allocated.
    // <already included in the C.2 case above>
    // C.7
    //   If the argument is an Integral or Pointer Type, the size of the
    //   argument is less than or equal to 8 bytes and the NGRN is less than 8,
    //   the argument is copied to the least significant bits in x[NGRN].
    //   The NGRN is incremented by one. The argument has now been allocated.
    // Here we treat any bitstype of the right size as integers or pointers
    // This is needed for types like Cstring which should be treated as
    // pointers. We don't need to worry about floating points here since they
    // are handled above.
    if (jl_is_immutable(dt) && jl_datatype_nfields(dt) == 0 &&
        (dt->size == 1 || dt->size == 2 || dt->size == 4 ||
         dt->size == 8 || dt->size == 16))
        return;

    // C.8
    //   If the argument has an alignment of 16 then the NGRN is rounded up to
    //   the next even number.
    // C.9
    //   If the argument is an Integral Type, the size of the argument is equal
    //   to 16 and the NGRN is less than 7, the argument is copied to x[NGRN]
    //   and x[NGRN+1]. x[NGRN] shall contain the lower addressed double-word
    //   of the memory representation of the argument. The NGRN is incremented
    //   by two. The argument has now been allocated.
    // <merged into C.7 above>
    // C.10
    //   If the argument is a Composite Type and the size in double-words of
    //   the argument is not more than 8 minus NGRN, then the argument is
    //   copied into consecutive general-purpose registers, starting at x[NGRN].
    //   The argument is passed as though it had been loaded into the registers
    //   from a double-word-aligned address with an appropriate sequence of LDR
    //   instructions loading consecutive registers from memory (the contents of
    //   any unused parts of the registers are unspecified by this standard).
    //   The NGRN is incremented by the number of registers used. The argument
    //   has now been allocated.
    // We don't check for composite types here since the ones that have
    // corresponding C types are already handled and we just treat the ones
    // with weird size as a black box composite type.
    // The type can fit in 8 x 8 bytes since it is handled by
    // need_pass_by_ref otherwise.
    *need_rewrite = true;

    // C.11
    //   The NGRN is set to 8.
    // C.12
    //   The NSAA is rounded up to the larger of 8 or the Natural Alignment
    //   of the argument’s type.
    // C.13
    //   If the argument is a composite type then the argument is copied to
    //   memory at the adjusted NSAA. The NSAA is incremented by the size of
    //   the argument. The argument has now been allocated.
    // <handled by C.10 above>
    // C.14
    //   If the size of the argument is less than 8 bytes then the size of the
    //   argument is set to 8 bytes. The effect is as if the argument was
    //   copied to the least significant bits of a 64-bit register and the
    //   remaining bits filled with unspecified values.
    // C.15
    //   The argument is copied to memory at the adjusted NSAA. The NSAA is
    //   incremented by the size of the argument. The argument has now been
    //   allocated.
    // <handled by C.10 above>
}

bool use_sret(AbiState*, jl_value_t *ty)
{
    // Assume jl_is_datatype(ty) && !jl_is_abstracttype(ty)
    // Section 5.5
    // If the type, T, of the result of a function is such that
    //
    //     void func(T arg)
    //
    // would require that arg be passed as a value in a register (or set of
    // registers) according to the rules in section 5.4 Parameter Passing,
    // then the result is returned in the same registers as would be used for
    // such an argument.
    bool fpreg = false;
    bool onstack = false;
    bool need_rewrite = false;
    classify_arg(ty, &fpreg, &onstack, &need_rewrite);
    return onstack;
}

Type *preferred_llvm_type(jl_value_t *ty, bool)
{
    if (!jl_is_datatype(ty) || jl_is_abstracttype(ty))
        return NULL;
    jl_datatype_t *dt = (jl_datatype_t*)ty;
    if (Type *fptype = get_llvm_fptype(dt))
        return fptype;
    bool fpreg = false;
    bool onstack = false;
    bool need_rewrite = false;
    classify_arg(ty, &fpreg, &onstack, &need_rewrite);
    if (!need_rewrite)
        return NULL;
    if (fpreg) {
        // Rewrite to [n x fptype] where n is the number of field
        // This only happens for isHFAorHVA
        size_t members = jl_datatype_nfields(dt);
        assert(members > 0 && members <= 4);
        jl_datatype_t *eltype = (jl_datatype_t*)jl_field_type(dt, 0);
        return ArrayType::get(get_llvm_fptype(eltype), members);
    }
    else {
        // Rewrite to [n x Int64] where n is the **size in dword**
        assert(dt->size <= 16); // Should be pass by reference otherwise
        return ArrayType::get(T_int64, (dt->size + 7) >> 3);
    }
}

}
