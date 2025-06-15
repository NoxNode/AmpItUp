## python ptx_gen.py && ptxas -O0 generated.ptx -o generated.cubin -arch=sm_86
## cuobjdump --dump-sass generated.cubin
## cuobjdump --dump-sass generated2.cubin
## xxd generated.cubin
import itertools

# PTX types
all_types = ["u32", "u64", "s32", "s64", "f32", "f64"]
b_types = ["b32"]
flt_types = ["f32", "f64"]
int_types = ["u32", "u64", "s32", "s64"]
shift_left_types = ["b32"]  # Only b32 is valid for shl
shift_right_types = ["b32", "u32", "s32"]  # Valid types for shr
u_types = ["u32", "u64"]
s_types = ["s32", "s64"]

# Arithmetic/Logic operations
alu_ops = ["add", "sub"]            # valid on int + float
b_ops = ["and", "or", "xor"]          # valid on b32 only
flt_ops = ["mul", "div.rn"]          # valid on float only
int_rem_ops = ["rem"]                # valid on int only

# Memory address spaces
mem_spaces = ["global", "shared", "const", "local"]

# Comparison operations
cmp_ops = ["eq", "ne", "lt", "le", "gt", "ge"]

# Branch operations (PTX uses setp + @predicate bra)
branch_ops = ["bra", "call", "ret", "exit"]

# Generate PTX preamble
ptx = [".version 7.5", ".target sm_86", ".address_size 64", ""]
ptx.append(".entry main() {")

# Declare registers
for ty in set(all_types + b_types):
    for i in range(4):
        #reg_prefix = "rb" if ty in b_types else "r"
        reg_prefix = "r" if ty in all_types + b_types else "b"
        ptx.append(f"    .reg .{ty} {reg_prefix}{ty}_{i};")
ptx.append("    .reg .u64 addr_u64;")
ptx.append("    .reg .pred p0;")
ptx.append("")

# ALU ops (add, sub) on int + float
for op in alu_ops:
    for ty in int_types + flt_types:
        for i in range(2):
            ptx.append(f"    {op}.{ty} r{ty}_{i}, r{ty}_{i}, r{ty}_{i};")

# Bitwise ops only on b32
for op in b_ops:
    for i in range(2):
        ptx.append(f"    {op}.b32 rb32_{i}, rb32_{i}, rb32_{i};")

# Shift left ops (only b32)
for ty in shift_left_types:
    for i in range(2):
        ptx.append(f"    shl.{ty} r{ty}_{i}, r{ty}_{i}, r{ty}_{i};")

# Shift right ops (b32, u32, s32)
for ty in shift_right_types:
    for i in range(2):
        ptx.append(f"    shr.{ty} r{ty}_{i}, r{ty}_{i}, r{ty}_{i};")

# Float ops: mul, div.rn
for op in flt_ops:
    for ty in flt_types:
        for i in range(2):
            ptx.append(f"    {op}.{ty} r{ty}_{i}, r{ty}_{i}, r{ty}_{i};")

# Int-only ops: rem
for op in int_rem_ops:
    for ty in int_types:
        for i in range(2):
            ptx.append(f"    {op}.{ty} r{ty}_{i}, r{ty}_{i}, r{ty}_{i};")

# Memory ops with proper u64 address register
for space in mem_spaces:
    for ty in all_types:
        ptx.append(f"    ld.{space}.{ty} r{ty}_0, [addr_u64];")
        if space != "const": ptx.append(f"    st.{space}.{ty} [addr_u64], r{ty}_0;")

# Comparison ops
for cmp in cmp_ops:
    for ty in all_types:
        ptx.append(f"    setp.{cmp}.{ty} p0, r{ty}_0, r{ty}_1;")

# Branching
ptx.append("    @p0 bra target_label;")

# Dummy label and function
ptx.append("target_label:")
ptx.append("    add.u32 ru32_0, ru32_0, ru32_0;")
ptx.append("    ret;")
ptx.append("    exit;")
ptx.append("")

ptx.append("}")

# Write to file
with open("generated.ptx", "w") as f:
    f.write("\n".join(ptx))

print("PTX file 'generated.ptx' created.")


"""



                                                                     s2 s1 dest
                                                                      -v-v-v
IADD3 R2, R2, R2, RZ ;                                     /* 0x0000000202027210 */
                                                           /* 0x003fde0007ffe0ff */
                                                                              -^
                                                                              s3
IADD3 R4, P0, R4, R4, RZ ;                                 /* 0x0000000404047210 */
                                                           /* 0x003fde0007f1e0ff */
IADD3.X R5, R5, R5, RZ, P0, !PT ;                          /* 0x0000000505057210 */
                                                           /* 0x003fde00007fe4ff */

IADD3 R6, P1, R5, -R7, RZ ;                                /* 0x8000000705067210 */
                                                           /* 0x003fde0007f3e0ff */
IADD3.X R7, R9, ~R8, RZ, P1, !PT ;                         /* 0x8000000809077210 */
                                                           /* 0x003fde0000ffe4ff */

FADD R22, R0, R0 ;                                         /* 0x0000000000167221 */
                                                           /* 0x003fde0000000000 */

                                                                        -v-v
DADD R16, R16, R16 ;                                       /* 0x0000000010107229 */
                                                           /* 0x00321e0000000010 */
                                                                              -^
                                                                               dest

DADD R16, R16, -R16 ;                                      /* 0x0000000010107229 */
                                                           /* 0x00321e0000000810 */
                                                                             ^ negate src2 bit


                                                                addr_off>>2
                                                                    -v
MOV R1, c[0x0][0x28] ;                                     /* 0x00000a0000017a02 */
                                                           /* 0x003fde0000000f00 */

        /*3e30*/                   IADD3 R7, P1, -R3, RZ, RZ ;                                /* 0x000000ff03077210 */
                                                                                              /* 0x003fde0007f3e1ff */
        /*3e40*/                   IADD3.X R8, ~R4, RZ, RZ, P1, !PT ;                         /* 0x000000ff04087210 */
                                                                                              /* 0x003fde0000ffe5ff */
        /*3d70*/                   IADD3 R6, P0, -R8, RZ, RZ ;                                /* 0x000000ff08067210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*3d80*/                   IADD3.X R7, ~R2, RZ, RZ, P0, !PT ;                         /* 0x000000ff02077210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*4f60*/                   IADD3.X RZ, P0, R20, R24, RZ, P0, !PT ;                    /* 0x0000001814ff7210 */
                                                                                              /* 0x003fde000071e4ff */

	code for sm_86
		Function : main
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM86 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM86)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                                     /* 0x00000a0000017a02 */
                                                                                              /* 0x003fde0000000f00 */
        /*0010*/                   MOV R14, c[0x0][0x118] ;                                   /* 0x00004600000e7a02 */
                                                                                              /* 0x003fde0000000f00 */
        /*0020*/                   MOV R15, c[0x0][0x11c] ;                                   /* 0x00004700000f7a02 */
                                                                                              /* 0x003fde0000000f00 */
        /*0030*/                   IADD3 R2, R2, R2, RZ ;                                     /* 0x0000000202027210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*0040*/                   IADD3 R3, R3, R3, RZ ;                                     /* 0x0000000303037210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*0050*/                   IADD3 R4, P0, R4, R4, RZ ;                                 /* 0x0000000404047210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*0060*/                   IADD3.X R5, R5, R5, RZ, P0, !PT ;                          /* 0x0000000505057210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*0070*/                   IADD3 R6, P0, R6, R6, RZ ;                                 /* 0x0000000606067210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*0080*/                   IADD3.X R7, R7, R7, RZ, P0, !PT ;                          /* 0x0000000707077210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*0090*/                   IADD3 R8, R8, R8, RZ ;                                     /* 0x0000000808087210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*00a0*/                   IADD3 R9, R9, R9, RZ ;                                     /* 0x0000000909097210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*00b0*/                   IADD3 R10, P0, R10, R10, RZ ;                              /* 0x0000000a0a0a7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*00c0*/                   IADD3.X R11, R11, R11, RZ, P0, !PT ;                       /* 0x0000000b0b0b7210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*00d0*/                   IADD3 R12, P0, R12, R12, RZ ;                              /* 0x0000000c0c0c7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*00e0*/                   IADD3.X R13, R13, R13, RZ, P0, !PT ;                       /* 0x0000000d0d0d7210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*00f0*/                   FADD R25, R25, R25 ;                                       /* 0x0000001919197221 */
                                                                                              /* 0x003fde0000000000 */
        /*0100*/                   FADD R22, R0, R0 ;                                         /* 0x0000000000167221 */
                                                                                              /* 0x003fde0000000000 */
        /*0110*/                   DADD R18, R18, R18 ;                                       /* 0x0000000012127229 */
                                                                                              /* 0x00321e0000000012 */
        /*0120*/                   DADD R16, R16, R16 ;                                       /* 0x0000000010107229 */
                                                                                              /* 0x00321e0000000010 */
        /*0130*/                   IADD3 R0, R2, -R2, RZ ;                                    /* 0x8000000202007210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*0140*/                   IADD3 R2, R3, -R3, RZ ;                                    /* 0x8000000303027210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*0150*/                   IADD3 R4, P0, R4, -R4, RZ ;                                /* 0x8000000404047210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*0160*/                   IADD3.X R5, R5, ~R5, RZ, P0, !PT ;                         /* 0x8000000505057210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*0170*/                   IADD3 R6, P0, R6, -R6, RZ ;                                /* 0x8000000606067210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*0180*/                   IADD3.X R7, R7, ~R7, RZ, P0, !PT ;                         /* 0x8000000707077210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*0190*/                   IADD3 R8, R8, -R8, RZ ;                                    /* 0x8000000808087210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*01a0*/                   IADD3 R9, R9, -R9, RZ ;                                    /* 0x8000000909097210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*01b0*/                   IADD3 R10, P0, R10, -R10, RZ ;                             /* 0x8000000a0a0a7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*01c0*/                   IADD3.X R11, R11, ~R11, RZ, P0, !PT ;                      /* 0x8000000b0b0b7210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*01d0*/                   IADD3 R12, P0, R12, -R12, RZ ;                             /* 0x8000000c0c0c7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*01e0*/                   IADD3.X R13, R13, ~R13, RZ, P0, !PT ;                      /* 0x8000000d0d0d7210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*01f0*/                   FADD R23, R25, -R25 ;                                      /* 0x8000001919177221 */
                                                                                              /* 0x003fde0000000000 */
        /*0200*/                   FADD R24, R22, -R22 ;                                      /* 0x8000001616187221 */
                                                                                              /* 0x003fde0000000000 */
        /*0210*/                   DADD R18, R18, -R18 ;                                      /* 0x0000000012127229 */
                                                                                              /* 0x00321e0000000812 */
        /*0220*/                   DADD R16, R16, -R16 ;                                      /* 0x0000000010107229 */
                                                                                              /* 0x00321e0000000810 */
        /*0230*/                   LOP3.LUT R21, R21, R21, RZ, 0xc0, !PT ;                    /* 0x0000001515157212 */
                                                                                              /* 0x003fde00078ec0ff */
        /*0240*/                   LOP3.LUT R20, R20, R20, RZ, 0xc0, !PT ;                    /* 0x0000001414147212 */
                                                                                              /* 0x003fde00078ec0ff */
        /*0250*/                   LOP3.LUT R21, R21, R21, RZ, 0xfc, !PT ;                    /* 0x0000001515157212 */
                                                                                              /* 0x003fde00078efcff */
        /*0260*/                   LOP3.LUT R20, R20, R20, RZ, 0xfc, !PT ;                    /* 0x0000001414147212 */
                                                                                              /* 0x003fde00078efcff */
        /*0270*/                   LOP3.LUT R21, R21, R21, RZ, 0x3c, !PT ;                    /* 0x0000001515157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*0280*/                   LOP3.LUT R20, R20, R20, RZ, 0x3c, !PT ;                    /* 0x0000001414147212 */
                                                                                              /* 0x003fde00078e3cff */
        /*0290*/                   SHF.L.U32 R21, R21, R21, RZ ;                              /* 0x0000001515157219 */
                                                                                              /* 0x003fde00000006ff */
        /*02a0*/                   SHF.L.U32 R20, R20, R20, RZ ;                              /* 0x0000001414147219 */
                                                                                              /* 0x003fde00000006ff */
        /*02b0*/                   SHF.R.U32.HI R21, RZ, R21, R21 ;                           /* 0x00000015ff157219 */
                                                                                              /* 0x003fde0000011615 */
        /*02c0*/                   SHF.R.U32.HI R20, RZ, R20, R20 ;                           /* 0x00000014ff147219 */
                                                                                              /* 0x003fde0000011614 */
        /*02d0*/                   SHF.R.U32.HI R25, RZ, R0, R0 ;                             /* 0x00000000ff197219 */
                                                                                              /* 0x003fde0000011600 */
        /*02e0*/                   SHF.R.U32.HI R3, RZ, R2, R2 ;                              /* 0x00000002ff037219 */
                                                                                              /* 0x003fde0000011602 */
        /*02f0*/                   SHF.R.S32.HI R8, RZ, R8, R8 ;                              /* 0x00000008ff087219 */
                                                                                              /* 0x003fde0000011408 */
        /*0300*/                   SHF.R.S32.HI R9, RZ, R9, R9 ;                              /* 0x00000009ff097219 */
                                                                                              /* 0x003fde0000011409 */
        /*0310*/                   FMUL R22, R23, R23 ;                                       /* 0x0000001717167220 */
                                                                                              /* 0x003fde0000400000 */
        /*0320*/                   FMUL R0, R24, R24 ;                                        /* 0x0000001818007220 */
                                                                                              /* 0x003fde0000400000 */
        /*0330*/                   DMUL R18, R18, R18 ;                                       /* 0x0000001212127228 */
                                                                                              /* 0x00321e0000000000 */
        /*0340*/                   DMUL R16, R16, R16 ;                                       /* 0x0000001010107228 */
                                                                                              /* 0x00321e0000000000 */
        /*0350*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0360*/                   MOV R23, R22 ;                                             /* 0x0000001600177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0370*/                   MOV R27, R22 ;                                             /* 0x00000016001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0380*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0390*/                   MOV R2, R23 ;                                              /* 0x0000001700027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*03a0*/                   MUFU.RCP R24, R2 ;                                         /* 0x0000000200187308 */
                                                                                              /* 0x00321e0000001000 */
        /*03b0*/                   FADD R2, -RZ, -R2 ;                                        /* 0x80000002ff027221 */
                                                                                              /* 0x003fde0000000100 */
        /*03c0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*03d0*/                   MOV R26, 0x3f800000 ;                                      /* 0x3f800000001a7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*03e0*/                   FFMA R26, R2, R24, R26 ;                                   /* 0x00000018021a7223 */
                                                                                              /* 0x003fde000000001a */
        /*03f0*/                   FCHK P0, R27, R23 ;                                        /* 0x000000171b007302 */
                                                                                              /* 0x00321e0000000000 */
        /*0400*/                   FFMA R26, R24, R26, R24 ;                                  /* 0x0000001a181a7223 */
                                                                                              /* 0x003fde0000000018 */
        /*0410*/                   MOV R24, RZ ;                                              /* 0x000000ff00187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0420*/                   MOV R27, R22 ;                                             /* 0x00000016001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0430*/                   FFMA R24, R27, R26, R24 ;                                  /* 0x0000001a1b187223 */
                                                                                              /* 0x003fde0000000018 */
        /*0440*/                   FFMA R2, R2, R24, R27 ;                                    /* 0x0000001802027223 */
                                                                                              /* 0x003fde000000001b */
        /*0450*/                   FFMA R24, R2, R26, R24 ;                                   /* 0x0000001a02187223 */
                                                                                              /* 0x003fde0000000018 */
        /*0460*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0470*/                   MOV R2, R25 ;                                              /* 0x0000001900027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0480*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0490*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04a0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04b0*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04c0*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04d0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04e0*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*04f0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0500*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0510*/                   MOV R12, R12 ;                                             /* 0x0000000c000c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0520*/                   MOV R13, R13 ;                                             /* 0x0000000d000d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0530*/                   MOV R25, R22 ;                                             /* 0x0000001600197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0540*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0550*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0560*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0570*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0580*/                   MOV R17, R17 ;                                             /* 0x0000001100117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0590*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*05a0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*05b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*05c0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*05d0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*05e0*/               @P0 BRA 0x600 ;                                                /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*05f0*/                   BRA 0x690 ;                                                /* 0x0000009000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*0600*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0610*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0620*/                   MOV R25, R22 ;                                             /* 0x0000001600197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0630*/                   MOV R22, R23 ;                                             /* 0x0000001700167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0640*/                   MOV R20, 0x660 ;                                           /* 0x0000066000147802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0650*/                   CALL.REL.NOINC 0x5780 ;                                    /* 0x0000512000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*0660*/                   MOV R20, R22 ;                                             /* 0x0000001600147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0670*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0680*/                   MOV R24, R20 ;                                             /* 0x0000001400187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0690*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06a0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06b0*/                   MOV R20, R0 ;                                              /* 0x0000000000147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06c0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06d0*/                   MOV R25, R20 ;                                             /* 0x0000001400197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06e0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*06f0*/                   MUFU.RCP R21, R20 ;                                        /* 0x0000001400157308 */
                                                                                              /* 0x00321e0000001000 */
        /*0700*/                   FADD R22, -RZ, -R20 ;                                      /* 0x80000014ff167221 */
                                                                                              /* 0x003fde0000000100 */
        /*0710*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0720*/                   MOV R23, 0x3f800000 ;                                      /* 0x3f80000000177802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0730*/                   FFMA R23, R22, R21, R23 ;                                  /* 0x0000001516177223 */
                                                                                              /* 0x003fde0000000017 */
        /*0740*/                   FCHK P0, R0, R25 ;                                         /* 0x0000001900007302 */
                                                                                              /* 0x00321e0000000000 */
        /*0750*/                   FFMA R23, R21, R23, R21 ;                                  /* 0x0000001715177223 */
                                                                                              /* 0x003fde0000000015 */
        /*0760*/                   MOV R21, RZ ;                                              /* 0x000000ff00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0770*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0780*/                   FFMA R21, R0, R23, R21 ;                                   /* 0x0000001700157223 */
                                                                                              /* 0x003fde0000000015 */
        /*0790*/                   FFMA R22, R22, R21, R0 ;                                   /* 0x0000001516167223 */
                                                                                              /* 0x003fde0000000000 */
        /*07a0*/                   FFMA R22, R22, R23, R21 ;                                  /* 0x0000001716167223 */
                                                                                              /* 0x003fde0000000015 */
        /*07b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*07c0*/                   MOV R25, R24 ;                                             /* 0x0000001800197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*07d0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*07e0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*07f0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0800*/               @P0 BRA 0x820 ;                                                /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*0810*/                   BRA 0x8b0 ;                                                /* 0x0000009000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*0820*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0830*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0840*/                   MOV R25, R0 ;                                              /* 0x0000000000197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0850*/                   MOV R22, R20 ;                                             /* 0x0000001400167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0860*/                   MOV R20, 0x880 ;                                           /* 0x0000088000147802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0870*/                   CALL.REL.NOINC 0x5780 ;                                    /* 0x00004f0000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*0880*/                   MOV R0, R22 ;                                              /* 0x0000001600007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0890*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08a0*/                   MOV R22, R0 ;                                              /* 0x0000000000167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08b0*/                   MOV R0, R22 ;                                              /* 0x0000001600007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08c0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08d0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08e0*/                   MOV R20, R18 ;                                             /* 0x0000001200147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*08f0*/                   MOV R21, R19 ;                                             /* 0x0000001300157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0900*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0910*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0920*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0930*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0940*/                   MOV R22, R20 ;                                             /* 0x0000001400167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0950*/                   MOV R22, R21 ;                                             /* 0x0000001500167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0960*/                   MOV R23, RZ ;                                              /* 0x000000ff00177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0970*/                   MUFU.RCP64H R22, R22 ;                                     /* 0x0000001600167308 */
                                                                                              /* 0x00321e0000001800 */
        /*0980*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0990*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*09a0*/                   MOV R26, 0x1 ;                                             /* 0x00000001001a7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*09b0*/                   MOV R26, R26 ;                                             /* 0x0000001a001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*09c0*/                   MOV R27, R22 ;                                             /* 0x00000016001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*09d0*/                   DADD R22, -RZ, -R20 ;                                      /* 0x00000000ff167229 */
                                                                                              /* 0x00321e0000000914 */
        /*09e0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*09f0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0a00*/                   MOV R24, 0x0 ;                                             /* 0x0000000000187802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0a10*/                   MOV R25, 0x3ff00000 ;                                      /* 0x3ff0000000197802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0a20*/                   DFMA R28, R22, R26, R24 ;                                  /* 0x0000001a161c722b */
                                                                                              /* 0x00321e0000000018 */
        /*0a30*/                   DFMA R28, R28, R28, R28 ;                                  /* 0x0000001c1c1c722b */
                                                                                              /* 0x00321e000000001c */
        /*0a40*/                   DFMA R28, R28, R26, R26 ;                                  /* 0x0000001a1c1c722b */
                                                                                              /* 0x00321e000000001a */
        /*0a50*/                   DFMA R24, R22, R28, R24 ;                                  /* 0x0000001c1618722b */
                                                                                              /* 0x00321e0000000018 */
        /*0a60*/                   DFMA R24, R24, R28, R28 ;                                  /* 0x0000001c1818722b */
                                                                                              /* 0x00321e000000001c */
        /*0a70*/                   DMUL R26, R18, R24 ;                                       /* 0x00000018121a7228 */
                                                                                              /* 0x00321e0000000000 */
        /*0a80*/                   DFMA R22, R22, R26, R18 ;                                  /* 0x0000001a1616722b */
                                                                                              /* 0x00321e0000000012 */
        /*0a90*/                   DFMA R22, R22, R24, R26 ;                                  /* 0x000000181616722b */
                                                                                              /* 0x00321e000000001a */
        /*0aa0*/                   MOV R24, R19 ;                                             /* 0x0000001300187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ab0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ac0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ad0*/                   FADD R24, -RZ, |R24| ;                                     /* 0x40000018ff187221 */
                                                                                              /* 0x003fde0000000100 */
        /*0ae0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0af0*/                   FSETP.LT.AND P0, PT, R24, 6.5827683646048100446e-37, PT ;  /* 0x036000001800780b */
                                                                                              /* 0x003fde0003f01000 */
        /*0b00*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b10*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b20*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b30*/                   MOV R25, R18 ;                                             /* 0x0000001200197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b40*/                   MOV R26, R19 ;                                             /* 0x00000013001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b50*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b60*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b70*/                   MOV R24, R20 ;                                             /* 0x0000001400187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b80*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0b90*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ba0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0bb0*/               @P0 BRA 0xc80 ;                                                /* 0x000000c000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*0bc0*/                   MOV R0, R21 ;                                              /* 0x0000001500007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0bd0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0be0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0bf0*/                   MOV R18, R23 ;                                             /* 0x0000001700127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c00*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c10*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c20*/                   MOV R19, RZ ;                                              /* 0x000000ff00137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c30*/                   FFMA R0, R19, R0, R18 ;                                    /* 0x0000000013007223 */
                                                                                              /* 0x003fde0000000012 */
        /*0c40*/                   FADD R0, -RZ, |R0| ;                                       /* 0x40000000ff007221 */
                                                                                              /* 0x003fde0000000100 */
        /*0c50*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c60*/                   FSETP.GT.AND P0, PT, R0, 1.469367938527859385e-39, PT ;    /* 0x001000000000780b */
                                                                                              /* 0x003fde0003f04000 */
        /*0c70*/               @P0 BRA 0xd20 ;                                                /* 0x000000a000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*0c80*/                   MOV R18, R25 ;                                             /* 0x0000001900127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0c90*/                   MOV R19, R26 ;                                             /* 0x0000001a00137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ca0*/                   MOV R20, R24 ;                                             /* 0x0000001800147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0cb0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0cc0*/                   MOV R0, 0xce0 ;                                            /* 0x00000ce000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0cd0*/                   CALL.REL.NOINC 0x2ba0 ;                                    /* 0x00001ec000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*0ce0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0cf0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d00*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d10*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d20*/                   MOV R18, R22 ;                                             /* 0x0000001600127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d30*/                   MOV R19, R23 ;                                             /* 0x0000001700137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d40*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d50*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d60*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d70*/                   MOV R17, R17 ;                                             /* 0x0000001100117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d80*/                   MOV R20, R16 ;                                             /* 0x0000001000147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0d90*/                   MOV R21, R17 ;                                             /* 0x0000001100157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0da0*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0db0*/                   MOV R17, R17 ;                                             /* 0x0000001100117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0dc0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0dd0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0de0*/                   MOV R0, R20 ;                                              /* 0x0000001400007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0df0*/                   MOV R0, R21 ;                                              /* 0x0000001500007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e00*/                   MOV R22, RZ ;                                              /* 0x000000ff00167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e10*/                   MUFU.RCP64H R0, R0 ;                                       /* 0x0000000000007308 */
                                                                                              /* 0x00321e0000001800 */
        /*0e20*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e30*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e40*/                   MOV R24, 0x1 ;                                             /* 0x0000000100187802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e50*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e60*/                   MOV R25, R0 ;                                              /* 0x0000000000197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e70*/                   DADD R22, -RZ, -R20 ;                                      /* 0x00000000ff167229 */
                                                                                              /* 0x00321e0000000914 */
        /*0e80*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0e90*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ea0*/                   MOV R26, 0x0 ;                                             /* 0x00000000001a7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0eb0*/                   MOV R27, 0x3ff00000 ;                                      /* 0x3ff00000001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ec0*/                   DFMA R28, R22, R24, R26 ;                                  /* 0x00000018161c722b */
                                                                                              /* 0x00321e000000001a */
        /*0ed0*/                   DFMA R28, R28, R28, R28 ;                                  /* 0x0000001c1c1c722b */
                                                                                              /* 0x00321e000000001c */
        /*0ee0*/                   DFMA R28, R28, R24, R24 ;                                  /* 0x000000181c1c722b */
                                                                                              /* 0x00321e0000000018 */
        /*0ef0*/                   DFMA R26, R22, R28, R26 ;                                  /* 0x0000001c161a722b */
                                                                                              /* 0x00321e000000001a */
        /*0f00*/                   DFMA R26, R26, R28, R28 ;                                  /* 0x0000001c1a1a722b */
                                                                                              /* 0x00321e000000001c */
        /*0f10*/                   DMUL R24, R16, R26 ;                                       /* 0x0000001a10187228 */
                                                                                              /* 0x00321e0000000000 */
        /*0f20*/                   DFMA R22, R22, R24, R16 ;                                  /* 0x000000181616722b */
                                                                                              /* 0x00321e0000000010 */
        /*0f30*/                   DFMA R22, R22, R26, R24 ;                                  /* 0x0000001a1616722b */
                                                                                              /* 0x00321e0000000018 */
        /*0f40*/                   MOV R0, R17 ;                                              /* 0x0000001100007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0f50*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0f60*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0f70*/                   FADD R0, -RZ, |R0| ;                                       /* 0x40000000ff007221 */
                                                                                              /* 0x003fde0000000100 */
        /*0f80*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0f90*/                   FSETP.LT.AND P0, PT, R0, 6.5827683646048100446e-37, PT ;   /* 0x036000000000780b */
                                                                                              /* 0x003fde0003f01000 */
        /*0fa0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0fb0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0fc0*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0fd0*/                   MOV R17, R17 ;                                             /* 0x0000001100117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0fe0*/                   MOV R18, R16 ;                                             /* 0x0000001000127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*0ff0*/                   MOV R19, R17 ;                                             /* 0x0000001100137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1000*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1010*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1020*/                   MOV R24, R20 ;                                             /* 0x0000001400187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1030*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1040*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1050*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1060*/               @P0 BRA 0x1130 ;                                               /* 0x000000c000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*1070*/                   MOV R0, R21 ;                                              /* 0x0000001500007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1080*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1090*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*10a0*/                   MOV R16, R23 ;                                             /* 0x0000001700107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*10b0*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*10c0*/                   MOV R16, R16 ;                                             /* 0x0000001000107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*10d0*/                   MOV R17, RZ ;                                              /* 0x000000ff00117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*10e0*/                   FFMA R0, R17, R0, R16 ;                                    /* 0x0000000011007223 */
                                                                                              /* 0x003fde0000000010 */
        /*10f0*/                   FADD R0, -RZ, |R0| ;                                       /* 0x40000000ff007221 */
                                                                                              /* 0x003fde0000000100 */
        /*1100*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1110*/                   FSETP.GT.AND P0, PT, R0, 1.469367938527859385e-39, PT ;    /* 0x001000000000780b */
                                                                                              /* 0x003fde0003f04000 */
        /*1120*/               @P0 BRA 0x11d0 ;                                               /* 0x000000a000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*1130*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1140*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1150*/                   MOV R20, R24 ;                                             /* 0x0000001800147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1160*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1170*/                   MOV R0, 0x1190 ;                                           /* 0x0000119000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*1180*/                   CALL.REL.NOINC 0x2ba0 ;                                    /* 0x00001a1000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*1190*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11a0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11c0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11d0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11e0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*11f0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1200*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1210*/                   MOV R0, R2 ;                                               /* 0x0000000200007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1220*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1230*/                   MOV R16, R22 ;                                             /* 0x0000001600107202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1240*/                   MOV R17, R23 ;                                             /* 0x0000001700117202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1250*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1260*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1270*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1280*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1290*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*12a0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*12b0*/                   I2F.U32.RP R18, R2 ;                                       /* 0x0000000200127306 */
                                                                                              /* 0x00321e0000209000 */
        /*12c0*/                   MUFU.RCP R18, R18 ;                                        /* 0x0000001200127308 */
                                                                                              /* 0x00321e0000001000 */
        /*12d0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*12e0*/                   IADD3 R18, R18, 0xffffffe, RZ ;                            /* 0x0ffffffe12127810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*12f0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1300*/                   F2I.FTZ.U32.TRUNC.NTZ R18, R18 ;                           /* 0x0000001200127305 */
                                                                                              /* 0x00321e000021f000 */
        /*1310*/                   IMAD.U32 R19, R18, R2, RZ ;                                /* 0x0000000212137224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1320*/                   IADD3 R19, RZ, -R19, RZ ;                                  /* 0x80000013ff137210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1330*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1340*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1350*/                   IMAD.HI.U32 R19, R18, R19, RZ ;                            /* 0x0000001312137227 */
                                                                                              /* 0x003fde00078e00ff */
        /*1360*/                   IADD3 R19, R18, R19, RZ ;                                  /* 0x0000001312137210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1370*/                   IMAD.HI.U32 R19, R19, R0, RZ ;                             /* 0x0000000013137227 */
                                                                                              /* 0x003fde00078e00ff */
        /*1380*/                   IMAD.U32 R19, R19, R2, RZ ;                                /* 0x0000000213137224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1390*/                   IADD3 R19, R0, -R19, RZ ;                                  /* 0x8000001300137210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*13a0*/                   ISETP.GE.U32.AND P0, PT, R19, R2, PT ;                     /* 0x000000021300720c */
                                                                                              /* 0x003fde0003f06070 */
        /*13b0*/                   IADD3 R0, R19, -R2, RZ ;                                   /* 0x8000000213007210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*13c0*/                   SEL R0, R0, R19, P0 ;                                      /* 0x0000001300007207 */
                                                                                              /* 0x003fde0000000000 */
        /*13d0*/                   IADD3 R18, R0, -R2, RZ ;                                   /* 0x8000000200127210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*13e0*/                   ISETP.GE.U32.AND P0, PT, R0, R2, PT ;                      /* 0x000000020000720c */
                                                                                              /* 0x003fde0003f06070 */
        /*13f0*/                   SEL R18, R18, R0, P0 ;                                     /* 0x0000000012127207 */
                                                                                              /* 0x003fde0000000000 */
        /*1400*/                   MOV R0, RZ ;                                               /* 0x000000ff00007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1410*/                   ISETP.NE.U32.AND P0, PT, R2, R0, PT ;                      /* 0x000000000200720c */
                                                                                              /* 0x003fde0003f05070 */
        /*1420*/                   LOP3.LUT R2, RZ, R2, RZ, 0x33, !PT ;                       /* 0x00000002ff027212 */
                                                                                              /* 0x003fde00078e33ff */
        /*1430*/                   PLOP3.LUT P0, PT, P0, PT, PT, 0x8, 0x0 ;                   /* 0x000000000000781c */
                                                                                              /* 0x003fde000070e170 */
        /*1440*/                   SEL R2, R2, R18, P0 ;                                      /* 0x0000001202027207 */
                                                                                              /* 0x003fde0000000000 */
        /*1450*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1460*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1470*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1480*/                   MOV R0, R3 ;                                               /* 0x0000000300007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1490*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14a0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14b0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14c0*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14d0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14e0*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*14f0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1500*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1510*/                   I2F.U32.RP R2, R3 ;                                        /* 0x0000000300027306 */
                                                                                              /* 0x00321e0000209000 */
        /*1520*/                   MUFU.RCP R2, R2 ;                                          /* 0x0000000200027308 */
                                                                                              /* 0x00321e0000001000 */
        /*1530*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1540*/                   IADD3 R2, R2, 0xffffffe, RZ ;                              /* 0x0ffffffe02027810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1550*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1560*/                   F2I.FTZ.U32.TRUNC.NTZ R2, R2 ;                             /* 0x0000000200027305 */
                                                                                              /* 0x00321e000021f000 */
        /*1570*/                   IMAD.U32 R18, R2, R3, RZ ;                                 /* 0x0000000302127224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1580*/                   IADD3 R18, RZ, -R18, RZ ;                                  /* 0x80000012ff127210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1590*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*15a0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*15b0*/                   IMAD.HI.U32 R18, R2, R18, RZ ;                             /* 0x0000001202127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*15c0*/                   IADD3 R18, R2, R18, RZ ;                                   /* 0x0000001202127210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*15d0*/                   IMAD.HI.U32 R18, R18, R0, RZ ;                             /* 0x0000000012127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*15e0*/                   IMAD.U32 R18, R18, R3, RZ ;                                /* 0x0000000312127224 */
                                                                                              /* 0x003fde00078e00ff */
        /*15f0*/                   IADD3 R18, R0, -R18, RZ ;                                  /* 0x8000001200127210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1600*/                   ISETP.GE.U32.AND P0, PT, R18, R3, PT ;                     /* 0x000000031200720c */
                                                                                              /* 0x003fde0003f06070 */
        /*1610*/                   IADD3 R0, R18, -R3, RZ ;                                   /* 0x8000000312007210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1620*/                   SEL R0, R0, R18, P0 ;                                      /* 0x0000001200007207 */
                                                                                              /* 0x003fde0000000000 */
        /*1630*/                   IADD3 R2, R0, -R3, RZ ;                                    /* 0x8000000300027210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1640*/                   ISETP.GE.U32.AND P0, PT, R0, R3, PT ;                      /* 0x000000030000720c */
                                                                                              /* 0x003fde0003f06070 */
        /*1650*/                   SEL R2, R2, R0, P0 ;                                       /* 0x0000000002027207 */
                                                                                              /* 0x003fde0000000000 */
        /*1660*/                   MOV R0, RZ ;                                               /* 0x000000ff00007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1670*/                   ISETP.NE.U32.AND P0, PT, R3, R0, PT ;                      /* 0x000000000300720c */
                                                                                              /* 0x003fde0003f05070 */
        /*1680*/                   LOP3.LUT R3, RZ, R3, RZ, 0x33, !PT ;                       /* 0x00000003ff037212 */
                                                                                              /* 0x003fde00078e33ff */
        /*1690*/                   PLOP3.LUT P0, PT, P0, PT, PT, 0x8, 0x0 ;                   /* 0x000000000000781c */
                                                                                              /* 0x003fde000070e170 */
        /*16a0*/                   SEL R2, R3, R2, P0 ;                                       /* 0x0000000203027207 */
                                                                                              /* 0x003fde0000000000 */
        /*16b0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*16c0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*16d0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*16e0*/                   MOV R18, R4 ;                                              /* 0x0000000400127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*16f0*/                   MOV R19, R5 ;                                              /* 0x0000000500137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1700*/                   MOV R20, R4 ;                                              /* 0x0000000400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1710*/                   MOV R21, R5 ;                                              /* 0x0000000500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1720*/                   MOV R3, R2 ;                                               /* 0x0000000200037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1730*/                   MOV R0, 0x1750 ;                                           /* 0x0000175000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*1740*/                   CALL.REL.NOINC 0x4af0 ;                                    /* 0x000033a000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*1750*/                   MOV R2, R4 ;                                               /* 0x0000000400027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1760*/                   MOV R3, R5 ;                                               /* 0x0000000500037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1770*/                   MOV R18, R6 ;                                              /* 0x0000000600127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1780*/                   MOV R19, R7 ;                                              /* 0x0000000700137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1790*/                   MOV R20, R6 ;                                              /* 0x0000000600147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17a0*/                   MOV R21, R7 ;                                              /* 0x0000000700157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17b0*/                   MOV R4, R2 ;                                               /* 0x0000000200047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17c0*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17d0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17e0*/                   MOV R5, R2 ;                                               /* 0x0000000200057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*17f0*/                   MOV R0, 0x1810 ;                                           /* 0x0000181000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*1800*/                   CALL.REL.NOINC 0x4af0 ;                                    /* 0x000032e000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*1810*/                   MOV R2, R4 ;                                               /* 0x0000000400027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1820*/                   MOV R3, R5 ;                                               /* 0x0000000500037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1830*/                   MOV R0, R8 ;                                               /* 0x0000000800007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1840*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1850*/                   MOV R6, R2 ;                                               /* 0x0000000200067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1860*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1870*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1880*/                   MOV R7, R2 ;                                               /* 0x0000000200077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1890*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18a0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18b0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18c0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18d0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18e0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*18f0*/                   IABS R2, R8 ;                                              /* 0x0000000800027213 */
                                                                                              /* 0x003fde0000000000 */
        /*1900*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1910*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1920*/                   I2F.U32.RP R3, R2 ;                                        /* 0x0000000200037306 */
                                                                                              /* 0x00321e0000209000 */
        /*1930*/                   MUFU.RCP R3, R3 ;                                          /* 0x0000000300037308 */
                                                                                              /* 0x00321e0000001000 */
        /*1940*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1950*/                   IADD3 R3, R3, 0xffffffe, RZ ;                              /* 0x0ffffffe03037810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1960*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1970*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R3 ;                             /* 0x0000000300037305 */
                                                                                              /* 0x00321e000021f000 */
        /*1980*/                   IMAD.U32 R4, R2, R3, RZ ;                                  /* 0x0000000302047224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1990*/                   IADD3 R4, RZ, -R4, RZ ;                                    /* 0x80000004ff047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*19a0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*19b0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*19c0*/                   IMAD.HI.U32 R4, R3, R4, RZ ;                               /* 0x0000000403047227 */
                                                                                              /* 0x003fde00078e00ff */
        /*19d0*/                   IADD3 R4, R3, R4, RZ ;                                     /* 0x0000000403047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*19e0*/                   IABS R3, R0 ;                                              /* 0x0000000000037213 */
                                                                                              /* 0x003fde0000000000 */
        /*19f0*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1a00*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1a10*/                   IMAD.HI.U32 R4, R4, R3, RZ ;                               /* 0x0000000304047227 */
                                                                                              /* 0x003fde00078e00ff */
        /*1a20*/                   IMAD.U32 R4, R2, R4, RZ ;                                  /* 0x0000000402047224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1a30*/                   IADD3 R4, R3, -R4, RZ ;                                    /* 0x8000000403047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1a40*/                   ISETP.LE.U32.AND P0, PT, R2, R4, PT ;                      /* 0x000000040200720c */
                                                                                              /* 0x003fde0003f03070 */
        /*1a50*/                   IADD3 R3, R4, -R2, RZ ;                                    /* 0x8000000204037210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1a60*/                   SEL R3, R3, R4, P0 ;                                       /* 0x0000000403037207 */
                                                                                              /* 0x003fde0000000000 */
        /*1a70*/                   IADD3 R4, R3, -R2, RZ ;                                    /* 0x8000000203047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1a80*/                   ISETP.LE.U32.AND P0, PT, R2, R3, PT ;                      /* 0x000000030200720c */
                                                                                              /* 0x003fde0003f03070 */
        /*1a90*/                   SEL R4, R4, R3, P0 ;                                       /* 0x0000000304047207 */
                                                                                              /* 0x003fde0000000000 */
        /*1aa0*/                   MOV R2, RZ ;                                               /* 0x000000ff00027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ab0*/                   ISETP.GE.AND P0, PT, R0, R2, PT ;                          /* 0x000000020000720c */
                                                                                              /* 0x003fde0003f06270 */
        /*1ac0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ad0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ae0*/               @P0 BRA 0x1b30 ;                                               /* 0x0000004000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*1af0*/                   IADD3 R4, RZ, -R4, RZ ;                                    /* 0x80000004ff047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1b00*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1b10*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1b20*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1b30*/                   MOV R0, RZ ;                                               /* 0x000000ff00007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1b40*/                   ISETP.NE.AND P0, PT, R8, R0, PT ;                          /* 0x000000000800720c */
                                                                                              /* 0x003fde0003f05270 */
        /*1b50*/                   LOP3.LUT R8, RZ, R8, RZ, 0x33, !PT ;                       /* 0x00000008ff087212 */
                                                                                              /* 0x003fde00078e33ff */
        /*1b60*/                   PLOP3.LUT P0, PT, P0, PT, PT, 0x8, 0x0 ;                   /* 0x000000000000781c */
                                                                                              /* 0x003fde000070e170 */
        /*1b70*/                   SEL R4, R8, R4, P0 ;                                       /* 0x0000000408047207 */
                                                                                              /* 0x003fde0000000000 */
        /*1b80*/                   MOV R0, R4 ;                                               /* 0x0000000400007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1b90*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ba0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1bb0*/                   MOV R2, R9 ;                                               /* 0x0000000900027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1bc0*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1bd0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1be0*/                   MOV R8, R0 ;                                               /* 0x0000000000087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1bf0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c00*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c10*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c20*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c30*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c40*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c50*/                   IABS R0, R9 ;                                              /* 0x0000000900007213 */
                                                                                              /* 0x003fde0000000000 */
        /*1c60*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c70*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1c80*/                   I2F.U32.RP R3, R0 ;                                        /* 0x0000000000037306 */
                                                                                              /* 0x00321e0000209000 */
        /*1c90*/                   MUFU.RCP R3, R3 ;                                          /* 0x0000000300037308 */
                                                                                              /* 0x00321e0000001000 */
        /*1ca0*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1cb0*/                   IADD3 R3, R3, 0xffffffe, RZ ;                              /* 0x0ffffffe03037810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1cc0*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1cd0*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R3 ;                             /* 0x0000000300037305 */
                                                                                              /* 0x00321e000021f000 */
        /*1ce0*/                   IMAD.U32 R4, R0, R3, RZ ;                                  /* 0x0000000300047224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1cf0*/                   IADD3 R4, RZ, -R4, RZ ;                                    /* 0x80000004ff047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1d00*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1d10*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1d20*/                   IMAD.HI.U32 R4, R3, R4, RZ ;                               /* 0x0000000403047227 */
                                                                                              /* 0x003fde00078e00ff */
        /*1d30*/                   IADD3 R4, R3, R4, RZ ;                                     /* 0x0000000403047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1d40*/                   IABS R3, R2 ;                                              /* 0x0000000200037213 */
                                                                                              /* 0x003fde0000000000 */
        /*1d50*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1d60*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1d70*/                   IMAD.HI.U32 R4, R4, R3, RZ ;                               /* 0x0000000304047227 */
                                                                                              /* 0x003fde00078e00ff */
        /*1d80*/                   IMAD.U32 R4, R0, R4, RZ ;                                  /* 0x0000000400047224 */
                                                                                              /* 0x003fde00078e00ff */
        /*1d90*/                   IADD3 R4, R3, -R4, RZ ;                                    /* 0x8000000403047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1da0*/                   ISETP.LE.U32.AND P0, PT, R0, R4, PT ;                      /* 0x000000040000720c */
                                                                                              /* 0x003fde0003f03070 */
        /*1db0*/                   IADD3 R3, R4, -R0, RZ ;                                    /* 0x8000000004037210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1dc0*/                   SEL R3, R3, R4, P0 ;                                       /* 0x0000000403037207 */
                                                                                              /* 0x003fde0000000000 */
        /*1dd0*/                   IADD3 R4, R3, -R0, RZ ;                                    /* 0x8000000003047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1de0*/                   ISETP.LE.U32.AND P0, PT, R0, R3, PT ;                      /* 0x000000030000720c */
                                                                                              /* 0x003fde0003f03070 */
        /*1df0*/                   SEL R4, R4, R3, P0 ;                                       /* 0x0000000304047207 */
                                                                                              /* 0x003fde0000000000 */
        /*1e00*/                   MOV R0, RZ ;                                               /* 0x000000ff00007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e10*/                   ISETP.GE.AND P0, PT, R2, R0, PT ;                          /* 0x000000000200720c */
                                                                                              /* 0x003fde0003f06270 */
        /*1e20*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e30*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e40*/               @P0 BRA 0x1e90 ;                                               /* 0x0000004000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*1e50*/                   IADD3 R4, RZ, -R4, RZ ;                                    /* 0x80000004ff047210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*1e60*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e70*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e80*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1e90*/                   MOV R0, RZ ;                                               /* 0x000000ff00007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ea0*/                   ISETP.NE.AND P0, PT, R9, R0, PT ;                          /* 0x000000000900720c */
                                                                                              /* 0x003fde0003f05270 */
        /*1eb0*/                   LOP3.LUT R9, RZ, R9, RZ, 0x33, !PT ;                       /* 0x00000009ff097212 */
                                                                                              /* 0x003fde00078e33ff */
        /*1ec0*/                   PLOP3.LUT P0, PT, P0, PT, PT, 0x8, 0x0 ;                   /* 0x000000000000781c */
                                                                                              /* 0x003fde000070e170 */
        /*1ed0*/                   SEL R4, R9, R4, P0 ;                                       /* 0x0000000409047207 */
                                                                                              /* 0x003fde0000000000 */
        /*1ee0*/                   MOV R0, R4 ;                                               /* 0x0000000400007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ef0*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f00*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f10*/                   MOV R5, R10 ;                                              /* 0x0000000a00057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f20*/                   MOV R6, R11 ;                                              /* 0x0000000b00067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f30*/                   MOV R7, R10 ;                                              /* 0x0000000a00077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f40*/                   MOV R8, R11 ;                                              /* 0x0000000b00087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f50*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f60*/                   MOV R9, R0 ;                                               /* 0x0000000000097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f70*/                   MOV R0, 0x1f90 ;                                           /* 0x00001f9000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*1f80*/                   CALL.REL.NOINC 0x3cb0 ;                                    /* 0x00001d2000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*1f90*/                   MOV R2, R4 ;                                               /* 0x0000000400027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1fa0*/                   MOV R3, R5 ;                                               /* 0x0000000500037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1fb0*/                   MOV R5, R12 ;                                              /* 0x0000000c00057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1fc0*/                   MOV R6, R13 ;                                              /* 0x0000000d00067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1fd0*/                   MOV R7, R12 ;                                              /* 0x0000000c00077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1fe0*/                   MOV R8, R13 ;                                              /* 0x0000000d00087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*1ff0*/                   MOV R10, R2 ;                                              /* 0x00000002000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2000*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2010*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2020*/                   MOV R11, R2 ;                                              /* 0x00000002000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2030*/                   MOV R0, 0x2050 ;                                           /* 0x0000205000007802 */
                                                                                              /* 0x003fde0000000f00 */
        /*2040*/                   CALL.REL.NOINC 0x3cb0 ;                                    /* 0x00001c6000007944 */
                                                                                              /* 0x003fde0003c00000 */
        /*2050*/                   MOV R2, R4 ;                                               /* 0x0000000400027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2060*/                   MOV R3, R5 ;                                               /* 0x0000000500037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2070*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2080*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2090*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*20a0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*20b0*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*20c0*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*20d0*/                   LDG.E R6, [R4.64] ;                                        /* 0x0000000404067981 */
                                                                                              /* 0x00321e000c1e1900 */
        /*20e0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*20f0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2100*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2110*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2120*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2130*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2140*/                   STG.E [R4.64], R6 ;                                        /* 0x0000000604007986 */
                                                                                              /* 0x0033de000c101904 */
        /*2150*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2160*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2170*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2180*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2190*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*21a0*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*21b0*/                   LDG.E.64 R4, [R4.64] ;                                     /* 0x0000000404047981 */
                                                                                              /* 0x00321e000c1e1b00 */
        /*21c0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*21d0*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*21e0*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*21f0*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2200*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2210*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2220*/                   STG.E.64 [R6.64], R4 ;                                     /* 0x0000000406007986 */
                                                                                              /* 0x0033de000c101b04 */
        /*2230*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2240*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2250*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2260*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2270*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2280*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2290*/                   LDG.E R6, [R4.64] ;                                        /* 0x0000000404067981 */
                                                                                              /* 0x00321e000c1e1900 */
        /*22a0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*22b0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*22c0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*22d0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*22e0*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*22f0*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2300*/                   STG.E [R4.64], R6 ;                                        /* 0x0000000604007986 */
                                                                                              /* 0x0033de000c101904 */
        /*2310*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2320*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2330*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2340*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2350*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2360*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2370*/                   LDG.E.64 R4, [R4.64] ;                                     /* 0x0000000404047981 */
                                                                                              /* 0x00321e000c1e1b00 */
        /*2380*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2390*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*23a0*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*23b0*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*23c0*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*23d0*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*23e0*/                   STG.E.64 [R6.64], R4 ;                                     /* 0x0000000406007986 */
                                                                                              /* 0x0033de000c101b04 */
        /*23f0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2400*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2410*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2420*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2430*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2440*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2450*/                   LDG.E R6, [R4.64] ;                                        /* 0x0000000404067981 */
                                                                                              /* 0x00321e000c1e1900 */
        /*2460*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2470*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2480*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2490*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*24a0*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*24b0*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*24c0*/                   STG.E [R4.64], R6 ;                                        /* 0x0000000604007986 */
                                                                                              /* 0x0033de000c101904 */
        /*24d0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*24e0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*24f0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2500*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2510*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2520*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2530*/                   LDG.E.64 R4, [R4.64] ;                                     /* 0x0000000404047981 */
                                                                                              /* 0x00321e000c1e1b00 */
        /*2540*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2550*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2560*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2570*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2580*/                   R2UR UR4, R14 ;                                            /* 0x000000000e0473c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*2590*/                   R2UR UR5, R15 ;                                            /* 0x000000000f0573c2 */
                                                                                              /* 0x00321e00000e0000 */
        /*25a0*/                   STG.E.64 [R6.64], R4 ;                                     /* 0x0000000406007986 */
                                                                                              /* 0x0033de000c101b04 */
        /*25b0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*25c0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*25d0*/                   LDS R4, [R4] ;                                             /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000800 */
        /*25e0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*25f0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2600*/                   STS [R5], R4 ;                                             /* 0x0000000405007388 */
                                                                                              /* 0x0033de0000000800 */
        /*2610*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2620*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2630*/                   LDS.64 R4, [R4] ;                                          /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000a00 */
        /*2640*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2650*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2660*/                   STS.64 [R6], R4 ;                                          /* 0x0000000406007388 */
                                                                                              /* 0x0033de0000000a00 */
        /*2670*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2680*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2690*/                   LDS R4, [R4] ;                                             /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000800 */
        /*26a0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*26b0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*26c0*/                   STS [R5], R4 ;                                             /* 0x0000000405007388 */
                                                                                              /* 0x0033de0000000800 */
        /*26d0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*26e0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*26f0*/                   LDS.64 R4, [R4] ;                                          /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000a00 */
        /*2700*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2710*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2720*/                   STS.64 [R6], R4 ;                                          /* 0x0000000406007388 */
                                                                                              /* 0x0033de0000000a00 */
        /*2730*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2740*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2750*/                   LDS R4, [R4] ;                                             /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000800 */
        /*2760*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2770*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2780*/                   STS [R5], R4 ;                                             /* 0x0000000405007388 */
                                                                                              /* 0x0033de0000000800 */
        /*2790*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*27a0*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*27b0*/                   LDS.64 R4, [R4] ;                                          /* 0x0000000004047984 */
                                                                                              /* 0x00321e0000000a00 */
        /*27c0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*27d0*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*27e0*/                   STS.64 [R6], R4 ;                                          /* 0x0000000406007388 */
                                                                                              /* 0x0033de0000000a00 */
        /*27f0*/                   MOV R8, R0 ;                                               /* 0x0000000000087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2800*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2810*/                   LDL R8, [R8] ;                                             /* 0x0000000008087983 */
                                                                                              /* 0x00321e0000100800 */
        /*2820*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2830*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2840*/                   STL [R4], R8 ;                                             /* 0x0000000804007387 */
                                                                                              /* 0x0033de0000100800 */
        /*2850*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2860*/                   MOV R5, R0 ;                                               /* 0x0000000000057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2870*/                   LDL.64 R4, [R4] ;                                          /* 0x0000000004047983 */
                                                                                              /* 0x00321e0000100a00 */
        /*2880*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2890*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*28a0*/                   STL.64 [R6], R4 ;                                          /* 0x0000000406007387 */
                                                                                              /* 0x0033de0000100a00 */
        /*28b0*/                   MOV R9, R0 ;                                               /* 0x0000000000097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*28c0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*28d0*/                   LDL R9, [R9] ;                                             /* 0x0000000009097983 */
                                                                                              /* 0x00321e0000100800 */
        /*28e0*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*28f0*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2900*/                   STL [R6], R9 ;                                             /* 0x0000000906007387 */
                                                                                              /* 0x0033de0000100800 */
        /*2910*/                   MOV R6, R0 ;                                               /* 0x0000000000067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2920*/                   MOV R7, R0 ;                                               /* 0x0000000000077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2930*/                   LDL.64 R6, [R6] ;                                          /* 0x0000000006067983 */
                                                                                              /* 0x00321e0000100a00 */
        /*2940*/                   MOV R10, R0 ;                                              /* 0x00000000000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2950*/                   MOV R11, R0 ;                                              /* 0x00000000000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2960*/                   STL.64 [R10], R6 ;                                         /* 0x000000060a007387 */
                                                                                              /* 0x0033de0000100a00 */
        /*2970*/                   MOV R25, R0 ;                                              /* 0x0000000000197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2980*/                   MOV R10, R0 ;                                              /* 0x00000000000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2990*/                   LDL R25, [R25] ;                                           /* 0x0000000019197983 */
                                                                                              /* 0x00321e0000100800 */
        /*29a0*/                   MOV R10, R0 ;                                              /* 0x00000000000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*29b0*/                   MOV R11, R0 ;                                              /* 0x00000000000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*29c0*/                   STL [R10], R25 ;                                           /* 0x000000190a007387 */
                                                                                              /* 0x0033de0000100800 */
        /*29d0*/                   MOV R18, R0 ;                                              /* 0x0000000000127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*29e0*/                   MOV R10, R0 ;                                              /* 0x00000000000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*29f0*/                   LDL.64 R18, [R18] ;                                        /* 0x0000000012127983 */
                                                                                              /* 0x00321e0000100a00 */
        /*2a00*/                   MOV R10, R0 ;                                              /* 0x00000000000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a10*/                   MOV R0, R0 ;                                               /* 0x0000000000007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a20*/                   STL.64 [R10], R18 ;                                        /* 0x000000120a007387 */
                                                                                              /* 0x0033de0000100a00 */
        /*2a30*/                   DSETP.GE.AND P0, PT, R18, R16, PT ;                        /* 0x000000101200722a */
                                                                                              /* 0x00321e0003f06000 */
        /*2a40*/                   MOV R12, R2 ;                                              /* 0x00000002000c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a50*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a60*/                   MOV R12, R12 ;                                             /* 0x0000000c000c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a70*/                   MOV R13, R2 ;                                              /* 0x00000002000d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a80*/                   MOV R2, R8 ;                                               /* 0x0000000800027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2a90*/                   MOV R0, R4 ;                                               /* 0x0000000400007202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2aa0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ab0*/                   MOV R4, R0 ;                                               /* 0x0000000000047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ac0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ad0*/                   MOV R8, R9 ;                                               /* 0x0000000900087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ae0*/                   MOV R10, R6 ;                                              /* 0x00000006000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2af0*/                   MOV R6, R7 ;                                               /* 0x0000000700067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b00*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b10*/                   MOV R11, R6 ;                                              /* 0x00000006000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b20*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b30*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b40*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b50*/              @!P0 BRA 0x2b60 ;                                               /* 0x0000000000008947 */
                                                                                              /* 0x003fde0003800000 */
        /*2b60*/                   IADD3 R2, R2, R2, RZ ;                                     /* 0x0000000202027210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*2b70*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2b80*/                   EXIT ;                                                     /* 0x000000000000794d */
                                                                                              /* 0x003fde0003800000 */
        /*2b90*/                   EXIT ;                                                     /* 0x000000000000794d */
                                                                                              /* 0x003fde0003800000 */
        /*2ba0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2bb0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2bc0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2bd0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2be0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2bf0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c00*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c10*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c20*/                   MOV R25, R20 ;                                             /* 0x0000001400197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c30*/                   MOV R26, R21 ;                                             /* 0x00000015001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c40*/                   MOV R20, R26 ;                                             /* 0x0000001a00147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c50*/                   MOV R32, R20 ;                                             /* 0x0000001400207202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2c60*/                   LOP3.LUT R24, R32, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff0000020187812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*2c70*/                   LOP3.LUT R20, R32, 0x800fffff, RZ, 0xc0, !PT ;             /* 0x800fffff20147812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*2c80*/                   LOP3.LUT R20, R20, 0x3ff00000, RZ, 0xfc, !PT ;             /* 0x3ff0000014147812 */
                                                                                              /* 0x003fde00078efcff */
        /*2c90*/                   MOV R21, R25 ;                                             /* 0x0000001900157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ca0*/                   MOV R33, R21 ;                                             /* 0x0000001500217202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2cb0*/                   MOV R22, R33 ;                                             /* 0x0000002100167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2cc0*/                   MOV R23, R20 ;                                             /* 0x0000001400177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2cd0*/                   MOV R20, R32 ;                                             /* 0x0000002000147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ce0*/                   FADD R20, -RZ, |R20| ;                                     /* 0x40000014ff147221 */
                                                                                              /* 0x003fde0000000100 */
        /*2cf0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d00*/                   FSETP.GEU.AND P0, PT, R20, 1.469367938527859385e-39, PT ;  /* 0x001000001400780b */
                                                                                              /* 0x003fde0003f0e000 */
        /*2d10*/                   MOV R34, R24 ;                                             /* 0x0000001800227202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d20*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d30*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d40*/                   MOV R20, R25 ;                                             /* 0x0000001900147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d50*/                   MOV R21, R26 ;                                             /* 0x0000001a00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d60*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d70*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d80*/                   MOV R32, R32 ;                                             /* 0x0000002000207202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2d90*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2da0*/                   MOV R33, R33 ;                                             /* 0x0000002100217202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2db0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2dc0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2dd0*/                   MOV R34, R34 ;                                             /* 0x0000002200227202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2de0*/               @P0 BRA 0x2e60 ;                                               /* 0x0000007000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*2df0*/                   DMUL R22, R20, 8.98846567431157953865e+307 ;               /* 0x7fe0000014167828 */
                                                                                              /* 0x00321e0000000000 */
        /*2e00*/                   MOV R25, R23 ;                                             /* 0x0000001700197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e10*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e20*/                   LOP3.LUT R25, R25, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff0000019197812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*2e30*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e40*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e50*/                   MOV R34, R25 ;                                             /* 0x0000001900227202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e60*/                   MOV R25, R22 ;                                             /* 0x0000001600197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e70*/                   MOV R25, R23 ;                                             /* 0x0000001700197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e80*/                   MOV R30, RZ ;                                              /* 0x000000ff001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2e90*/                   MUFU.RCP64H R25, R25 ;                                     /* 0x0000001900197308 */
                                                                                              /* 0x00321e0000001800 */
        /*2ea0*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2eb0*/                   MOV R31, R25 ;                                             /* 0x00000019001f7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ec0*/                   DADD R22, -RZ, -R22 ;                                      /* 0x00000000ff167229 */
                                                                                              /* 0x00321e0000000916 */
        /*2ed0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ee0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2ef0*/                   MOV R35, R19 ;                                             /* 0x0000001300237202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2f00*/                   MOV R35, R35 ;                                             /* 0x0000002300237202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2f10*/                   LOP3.LUT R36, R35, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff0000023247812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*2f20*/                   ISETP.LT.U32.AND P0, PT, R36, R24, PT ;                    /* 0x000000182400720c */
                                                                                              /* 0x003fde0003f01070 */
        /*2f30*/                   MOV R24, 0x1ca00000 ;                                      /* 0x1ca0000000187802 */
                                                                                              /* 0x003fde0000000f00 */
        /*2f40*/                   SEL R24, R24, 0x63400000, P0 ;                             /* 0x6340000018187807 */
                                                                                              /* 0x003fde0000000000 */
        /*2f50*/                   LOP3.LUT R25, R35, 0x800fffff, RZ, 0xc0, !PT ;             /* 0x800fffff23197812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*2f60*/                   LOP3.LUT R24, R24, R25, RZ, 0xfc, !PT ;                    /* 0x0000001918187212 */
                                                                                              /* 0x003fde00078efcff */
        /*2f70*/                   MOV R37, R18 ;                                             /* 0x0000001200257202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2f80*/                   MOV R37, R37 ;                                             /* 0x0000002500257202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2f90*/                   MOV R25, R24 ;                                             /* 0x0000001800197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2fa0*/                   MOV R24, R37 ;                                             /* 0x0000002500187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2fb0*/                   MOV R26, R35 ;                                             /* 0x00000023001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2fc0*/                   FADD R26, -RZ, |R26| ;                                     /* 0x4000001aff1a7221 */
                                                                                              /* 0x003fde0000000100 */
        /*2fd0*/                   MOV R26, R26 ;                                             /* 0x0000001a001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*2fe0*/                   FSETP.GEU.AND P0, PT, R26, 1.469367938527859385e-39, PT ;  /* 0x001000001a00780b */
                                                                                              /* 0x003fde0003f0e000 */
        /*2ff0*/                   MOV R38, R36 ;                                             /* 0x0000002400267202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3000*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3010*/                   MOV R31, R31 ;                                             /* 0x0000001f001f7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3020*/                   MOV R26, R30 ;                                             /* 0x0000001e001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3030*/                   MOV R30, R31 ;                                             /* 0x0000001f001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3040*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3050*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3060*/                   MOV R35, R35 ;                                             /* 0x0000002300237202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3070*/                   MOV R36, R36 ;                                             /* 0x0000002400247202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3080*/                   MOV R37, R37 ;                                             /* 0x0000002500257202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3090*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*30a0*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*30b0*/                   MOV R38, R38 ;                                             /* 0x0000002600267202 */
                                                                                              /* 0x003fde0000000f00 */
        /*30c0*/               @P0 BRA 0x3230 ;                                               /* 0x0000016000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*30d0*/                   LOP3.LUT R26, R35, 0x80000000, RZ, 0xc0, !PT ;             /* 0x80000000231a7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*30e0*/                   LOP3.LUT R27, R32, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff00000201b7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*30f0*/                   ISETP.LT.U32.AND P0, PT, R36, R27, PT ;                    /* 0x0000001b2400720c */
                                                                                              /* 0x003fde0003f01070 */
        /*3100*/                   MOV R27, 0x1ca00000 ;                                      /* 0x1ca00000001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*3110*/                   SEL R27, R27, 0x63400000, P0 ;                             /* 0x634000001b1b7807 */
                                                                                              /* 0x003fde0000000000 */
        /*3120*/                   LOP3.LUT R26, R26, R27, RZ, 0xfc, !PT ;                    /* 0x0000001b1a1a7212 */
                                                                                              /* 0x003fde00078efcff */
        /*3130*/                   LOP3.LUT R26, R26, 0x100000, RZ, 0xfc, !PT ;               /* 0x001000001a1a7812 */
                                                                                              /* 0x003fde00078efcff */
        /*3140*/                   MOV R28, RZ ;                                              /* 0x000000ff001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3150*/                   MOV R28, R28 ;                                             /* 0x0000001c001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3160*/                   MOV R29, R26 ;                                             /* 0x0000001a001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3170*/                   DADD R28, -RZ, -R28 ;                                      /* 0x00000000ff1c7229 */
                                                                                              /* 0x00321e000000091c */
        /*3180*/                   MOV R28, R28 ;                                             /* 0x0000001c001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3190*/                   MOV R29, R29 ;                                             /* 0x0000001d001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*31a0*/                   MOV R26, 0x0 ;                                             /* 0x00000000001a7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*31b0*/                   MOV R27, 0x40000000 ;                                      /* 0x40000000001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*31c0*/                   DFMA R24, R24, R26, R28 ;                                  /* 0x0000001a1818722b */
                                                                                              /* 0x00321e000000001c */
        /*31d0*/                   MOV R26, R25 ;                                             /* 0x00000019001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*31e0*/                   MOV R26, R26 ;                                             /* 0x0000001a001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*31f0*/                   LOP3.LUT R26, R26, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff000001a1a7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*3200*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3210*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3220*/                   MOV R38, R26 ;                                             /* 0x0000001a00267202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3230*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3240*/                   MOV R26, R30 ;                                             /* 0x0000001e001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3250*/                   MOV R27, 0x1 ;                                             /* 0x00000001001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*3260*/                   LOP3.LUT R27, R27, R26, RZ, 0x3c, !PT ;                    /* 0x0000001a1b1b7212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3270*/                   LOP3.LUT R26, R27, R26, RZ, 0x3c, !PT ;                    /* 0x0000001a1b1a7212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3280*/                   LOP3.LUT R27, R27, R26, RZ, 0x3c, !PT ;                    /* 0x0000001a1b1b7212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3290*/                   MOV R28, 0x0 ;                                             /* 0x00000000001c7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*32a0*/                   MOV R29, 0x3ff00000 ;                                      /* 0x3ff00000001d7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*32b0*/                   DFMA R30, R22, R26, R28 ;                                  /* 0x0000001a161e722b */
                                                                                              /* 0x00321e000000001c */
        /*32c0*/                   DFMA R30, R30, R30, R30 ;                                  /* 0x0000001e1e1e722b */
                                                                                              /* 0x00321e000000001e */
        /*32d0*/                   DFMA R26, R30, R26, R26 ;                                  /* 0x0000001a1e1a722b */
                                                                                              /* 0x00321e000000001a */
        /*32e0*/                   DFMA R28, R22, R26, R28 ;                                  /* 0x0000001a161c722b */
                                                                                              /* 0x00321e000000001c */
        /*32f0*/                   DFMA R26, R28, R26, R26 ;                                  /* 0x0000001a1c1a722b */
                                                                                              /* 0x00321e000000001a */
        /*3300*/                   DMUL R28, R26, R24 ;                                       /* 0x000000181a1c7228 */
                                                                                              /* 0x00321e0000000000 */
        /*3310*/                   DFMA R30, R22, R28, R24 ;                                  /* 0x0000001c161e722b */
                                                                                              /* 0x00321e0000000018 */
        /*3320*/                   DFMA R26, R30, R26, R28 ;                                  /* 0x0000001a1e1a722b */
                                                                                              /* 0x00321e000000001c */
        /*3330*/                   IADD3 R28, R38, -0x1, RZ ;                                 /* 0xffffffff261c7810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3340*/                   ISETP.GT.U32.AND P0, PT, R28, 0x7feffffe, PT ;             /* 0x7feffffe1c00780c */
                                                                                              /* 0x003fde0003f04070 */
        /*3350*/                   IADD3 R28, R34, -0x1, RZ ;                                 /* 0xffffffff221c7810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3360*/                   ISETP.GT.U32.AND P1, PT, R28, 0x7feffffe, PT ;             /* 0x7feffffe1c00780c */
                                                                                              /* 0x003fde0003f24070 */
        /*3370*/                   PLOP3.LUT P0, PT, P1, P0, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000f01570 */
        /*3380*/                   MOV R26, R26 ;                                             /* 0x0000001a001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3390*/                   MOV R27, R27 ;                                             /* 0x0000001b001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*33a0*/               @P0 BRA 0x33c0 ;                                               /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*33b0*/                   BRA 0x3560 ;                                               /* 0x000001a000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*33c0*/                   DSETP.NAN.AND P0, PT, R18, R18, PT ;                       /* 0x000000121200722a */
                                                                                              /* 0x00321e0003f08000 */
        /*33d0*/               @P0 BRA 0x3b00 ;                                               /* 0x0000072000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*33e0*/                   DSETP.NAN.AND P0, PT, R20, R20, PT ;                       /* 0x000000141400722a */
                                                                                              /* 0x00321e0003f08000 */
        /*33f0*/               @P0 BRA 0x3b60 ;                                               /* 0x0000076000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*3400*/                   ISETP.EQ.AND P0, PT, R38, R34, PT ;                        /* 0x000000222600720c */
                                                                                              /* 0x003fde0003f02270 */
        /*3410*/                   MOV R18, 0x0 ;                                             /* 0x0000000000127802 */
                                                                                              /* 0x003fde0000000f00 */
        /*3420*/                   MOV R19, 0xfff80000 ;                                      /* 0xfff8000000137802 */
                                                                                              /* 0x003fde0000000f00 */
        /*3430*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3440*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3450*/               @P0 BRA 0x3c20 ;                                               /* 0x000007c000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*3460*/                   LOP3.LUT R32, R35, R32, RZ, 0x3c, !PT ;                    /* 0x0000002023207212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3470*/                   LOP3.LUT R32, R32, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000020207812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*3480*/                   ISETP.EQ.AND P0, PT, R34, RZ, PT ;                         /* 0x000000ff2200720c */
                                                                                              /* 0x003fde0003f02270 */
        /*3490*/                   ISETP.EQ.AND P1, PT, R38, 0x7ff00000, PT ;                 /* 0x7ff000002600780c */
                                                                                              /* 0x003fde0003f22270 */
        /*34a0*/                   PLOP3.LUT P0, PT, P0, P1, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000703570 */
        /*34b0*/                   MOV R32, R32 ;                                             /* 0x0000002000207202 */
                                                                                              /* 0x003fde0000000f00 */
        /*34c0*/               @P0 BRA 0x34e0 ;                                               /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*34d0*/                   BRA 0x3bc0 ;                                               /* 0x000006e000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*34e0*/                   LOP3.LUT R18, R32, 0x7ff00000, RZ, 0xfc, !PT ;             /* 0x7ff0000020127812 */
                                                                                              /* 0x003fde00078efcff */
        /*34f0*/                   MOV R19, RZ ;                                              /* 0x000000ff00137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3500*/                   LOP3.LUT R19, R19, R18, RZ, 0x3c, !PT ;                    /* 0x0000001213137212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3510*/                   LOP3.LUT R18, R19, R18, RZ, 0x3c, !PT ;                    /* 0x0000001213127212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3520*/                   LOP3.LUT R19, R19, R18, RZ, 0x3c, !PT ;                    /* 0x0000001213137212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3530*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3540*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3550*/                   BRA 0x3c20 ;                                               /* 0x000006c000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*3560*/                   LOP3.LUT R19, R32, 0x7ff00000, RZ, 0xc0, !PT ;             /* 0x7ff0000020137812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*3570*/                   IADD3 R18, R36, -R19, RZ ;                                 /* 0x8000001324127210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3580*/                   IMNMX R18, R18, -0x46a00000, !PT ;                         /* 0xb960000012127817 */
                                                                                              /* 0x003fde0007800200 */
        /*3590*/                   IMNMX R18, R18, 0x46a00000, PT ;                           /* 0x46a0000012127817 */
                                                                                              /* 0x003fde0003800200 */
        /*35a0*/                   ISETP.LT.U32.AND P0, PT, R36, R19, PT ;                    /* 0x000000132400720c */
                                                                                              /* 0x003fde0003f01070 */
        /*35b0*/                   MOV R19, 0x1ca00000 ;                                      /* 0x1ca0000000137802 */
                                                                                              /* 0x003fde0000000f00 */
        /*35c0*/                   SEL R19, R19, 0x63400000, P0 ;                             /* 0x6340000013137807 */
                                                                                              /* 0x003fde0000000000 */
        /*35d0*/                   IADD3 R28, R18, -R19, RZ ;                                 /* 0x80000013121c7210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*35e0*/                   IADD3 R18, R28, 0x7fe00000, RZ ;                           /* 0x7fe000001c127810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*35f0*/                   MOV R29, RZ ;                                              /* 0x000000ff001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3600*/                   MOV R19, R18 ;                                             /* 0x0000001200137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3610*/                   MOV R18, R29 ;                                             /* 0x0000001d00127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3620*/                   DMUL R20, R26, R18 ;                                       /* 0x000000121a147228 */
                                                                                              /* 0x00321e0000000000 */
        /*3630*/                   MOV R30, R21 ;                                             /* 0x00000015001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3640*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3650*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3660*/                   FADD R30, -RZ, |R30| ;                                     /* 0x4000001eff1e7221 */
                                                                                              /* 0x003fde0000000100 */
        /*3670*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3680*/                   FSETP.GTU.AND P0, PT, R30, 1.469367938527859385e-39, PT ;  /* 0x001000001e00780b */
                                                                                              /* 0x003fde0003f0c000 */
        /*3690*/                   MOV R28, R28 ;                                             /* 0x0000001c001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36a0*/                   MOV R29, R29 ;                                             /* 0x0000001d001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36b0*/                   MOV R30, R18 ;                                             /* 0x00000012001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36c0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36d0*/                   MOV R18, R30 ;                                             /* 0x0000001e00127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36e0*/                   MOV R30, R19 ;                                             /* 0x00000013001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*36f0*/                   MOV R18, R20 ;                                             /* 0x0000001400127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3700*/                   MOV R19, R21 ;                                             /* 0x0000001500137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3710*/               @P0 BRA 0x3c20 ;                                               /* 0x0000050000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*3720*/                   DFMA R22, R22, R26, R24 ;                                  /* 0x0000001a1616722b */
                                                                                              /* 0x00321e0000000018 */
        /*3730*/                   MOV R20, R22 ;                                             /* 0x0000001600147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3740*/                   MOV R20, R23 ;                                             /* 0x0000001700147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3750*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3760*/                   MOV R22, R20 ;                                             /* 0x0000001400167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3770*/                   LOP3.LUT R32, R22, R32, RZ, 0x3c, !PT ;                    /* 0x0000002016207212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3780*/                   LOP3.LUT R32, R32, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000020207812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*3790*/                   MOV R20, R30 ;                                             /* 0x0000001e00147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*37a0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*37b0*/                   LOP3.LUT R20, R32, R20, RZ, 0xfc, !PT ;                    /* 0x0000001420147212 */
                                                                                              /* 0x003fde00078efcff */
        /*37c0*/                   MOV R21, R20 ;                                             /* 0x0000001400157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*37d0*/                   MOV R20, R29 ;                                             /* 0x0000001d00147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*37e0*/                   DMUL.RP R20, R26, R20 ;                                    /* 0x000000141a147228 */
                                                                                              /* 0x00321e0000008000 */
        /*37f0*/                   MOV R23, R20 ;                                             /* 0x0000001400177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3800*/                   MOV R20, R21 ;                                             /* 0x0000001500147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3810*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3820*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3830*/                   LOP3.LUT R20, R20, R32, RZ, 0x3c, !PT ;                    /* 0x0000002014147212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3840*/                   MOV R21, R23 ;                                             /* 0x0000001700157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3850*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3860*/                   LOP3.LUT R21, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3870*/                   LOP3.LUT R20, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415147212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3880*/                   LOP3.LUT R21, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3890*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*38a0*/                   FSETP.EQ.AND P0, PT, R22, RZ, PT ;                         /* 0x000000ff1600720b */
                                                                                              /* 0x003fde0003f02000 */
        /*38b0*/                   MOV R23, R20 ;                                             /* 0x0000001400177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*38c0*/                   MOV R22, R21 ;                                             /* 0x0000001500167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*38d0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*38e0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*38f0*/               @P0 BRA 0x3c20 ;                                               /* 0x0000032000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*3900*/                   IADD3 R20, RZ, -R28, RZ ;                                  /* 0x8000001cff147210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3910*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3920*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3930*/                   MOV R21, RZ ;                                              /* 0x000000ff00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3940*/                   LOP3.LUT R21, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3950*/                   LOP3.LUT R20, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415147212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3960*/                   LOP3.LUT R21, R21, R20, RZ, 0x3c, !PT ;                    /* 0x0000001415157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*3970*/                   DADD R24, -RZ, -R20 ;                                      /* 0x00000000ff187229 */
                                                                                              /* 0x00321e0000000914 */
        /*3980*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3990*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*39a0*/                   DFMA R24, R24, R18, R26 ;                                  /* 0x000000121818722b */
                                                                                              /* 0x00321e000000001a */
        /*39b0*/                   MOV R26, R24 ;                                             /* 0x00000018001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*39c0*/                   MOV R24, R25 ;                                             /* 0x0000001900187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*39d0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*39e0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*39f0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a00*/                   FADD R24, -RZ, |R24| ;                                     /* 0x40000018ff187221 */
                                                                                              /* 0x003fde0000000100 */
        /*3a10*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a20*/                   MOV R20, R21 ;                                             /* 0x0000001500147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a30*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a40*/                   IADD3 R20, R20, -0x43300000, RZ ;                          /* 0xbcd0000014147810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3a50*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a60*/                   FSETP.EQ.AND P0, PT, R24, R20, PT ;                        /* 0x000000141800720b */
                                                                                              /* 0x003fde0003f02000 */
        /*3a70*/                   MOV R20, R18 ;                                             /* 0x0000001200147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a80*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3a90*/                   FSEL R18, R23, R20, P0 ;                                   /* 0x0000001417127208 */
                                                                                              /* 0x003fde0000000000 */
        /*3aa0*/                   FSEL R19, R22, R19, P0 ;                                   /* 0x0000001316137208 */
                                                                                              /* 0x003fde0000000000 */
        /*3ab0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ac0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ad0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ae0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3af0*/                   BRA 0x3c20 ;                                               /* 0x0000012000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*3b00*/                   LOP3.LUT R18, R35, 0x80000, RZ, 0xfc, !PT ;                /* 0x0008000023127812 */
                                                                                              /* 0x003fde00078efcff */
        /*3b10*/                   MOV R19, R18 ;                                             /* 0x0000001200137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b20*/                   MOV R18, R37 ;                                             /* 0x0000002500127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b30*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b40*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b50*/                   BRA 0x3c20 ;                                               /* 0x000000c000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*3b60*/                   LOP3.LUT R18, R32, 0x80000, RZ, 0xfc, !PT ;                /* 0x0008000020127812 */
                                                                                              /* 0x003fde00078efcff */
        /*3b70*/                   MOV R19, R18 ;                                             /* 0x0000001200137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b80*/                   MOV R18, R33 ;                                             /* 0x0000002100127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3b90*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ba0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3bb0*/                   BRA 0x3c20 ;                                               /* 0x0000006000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*3bc0*/                   MOV R18, RZ ;                                              /* 0x000000ff00127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3bd0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3be0*/                   MOV R19, R32 ;                                             /* 0x0000002000137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3bf0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c00*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c10*/                   BRA 0x3c20 ;                                               /* 0x0000000000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*3c20*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c30*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c40*/                   MOV R22, R18 ;                                             /* 0x0000001200167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c50*/                   MOV R23, R19 ;                                             /* 0x0000001300177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c60*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c70*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c80*/                   MOV R18, R0 ;                                              /* 0x0000000000127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3c90*/                   MOV R19, 0x0 ;                                             /* 0x0000000000137802 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ca0*/                   RET.REL.NODEC R18 0x0 ;                                    /* 0xffffc35012007950 */
                                                                                              /* 0x003fde0003c3ffff */
        /*3cb0*/                   MOV R2, R5 ;                                               /* 0x0000000500027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3cc0*/                   MOV R3, R6 ;                                               /* 0x0000000600037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3cd0*/                   MOV R4, R7 ;                                               /* 0x0000000700047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ce0*/                   MOV R5, R8 ;                                               /* 0x0000000800057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3cf0*/                   MOV R8, R2 ;                                               /* 0x0000000200087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d00*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d10*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d20*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d30*/                   MOV R3, R4 ;                                               /* 0x0000000400037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d40*/                   MOV R4, R5 ;                                               /* 0x0000000500047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d50*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d60*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3d70*/                   IADD3 R6, P0, -R8, RZ, RZ ;                                /* 0x000000ff08067210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*3d80*/                   IADD3.X R7, ~R2, RZ, RZ, P0, !PT ;                         /* 0x000000ff02077210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*3d90*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3da0*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3db0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3dc0*/                   MOV R5, R2 ;                                               /* 0x0000000200057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3dd0*/                   ISETP.LT.AND P0, PT, R5, RZ, PT ;                          /* 0x000000ff0500720c */
                                                                                              /* 0x003fde0003f01270 */
        /*3de0*/                   SEL R6, R6, R8, P0 ;                                       /* 0x0000000806067207 */
                                                                                              /* 0x003fde0000000000 */
        /*3df0*/                   SEL R5, R7, R2, P0 ;                                       /* 0x0000000207057207 */
                                                                                              /* 0x003fde0000000000 */
        /*3e00*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3e10*/                   MOV R7, R4 ;                                               /* 0x0000000400077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3e20*/                   ISETP.LT.AND P0, PT, R7, RZ, PT ;                          /* 0x000000ff0700720c */
                                                                                              /* 0x003fde0003f01270 */
        /*3e30*/                   IADD3 R7, P1, -R3, RZ, RZ ;                                /* 0x000000ff03077210 */
                                                                                              /* 0x003fde0007f3e1ff */
        /*3e40*/                   IADD3.X R8, ~R4, RZ, RZ, P1, !PT ;                         /* 0x000000ff04087210 */
                                                                                              /* 0x003fde0000ffe5ff */
        /*3e50*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3e60*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3e70*/                   SEL R7, R7, R3, P0 ;                                       /* 0x0000000307077207 */
                                                                                              /* 0x003fde0000000000 */
        /*3e80*/                   SEL R8, R8, R4, P0 ;                                       /* 0x0000000408087207 */
                                                                                              /* 0x003fde0000000000 */
        /*3e90*/                   MOV R10, R7 ;                                              /* 0x00000007000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ea0*/                   MOV R11, R8 ;                                              /* 0x00000008000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3eb0*/                   I2F.U64.RP R10, R10 ;                                      /* 0x0000000a000a7312 */
                                                                                              /* 0x00321e0000309000 */
        /*3ec0*/                   MUFU.RCP R10, R10 ;                                        /* 0x0000000a000a7308 */
                                                                                              /* 0x00321e0000001000 */
        /*3ed0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ee0*/                   IADD3 R10, R10, 0x1ffffffe, RZ ;                           /* 0x1ffffffe0a0a7810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*3ef0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f00*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f10*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f20*/                   F2I.U64.TRUNC R10, R10 ;                                   /* 0x0000000a000a7311 */
                                                                                              /* 0x00321e000020d800 */
        /*3f30*/                   MOV R21, R10 ;                                             /* 0x0000000a00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f40*/                   MOV R22, R11 ;                                             /* 0x0000000b00167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f50*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f60*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f70*/                   MOV R9, R21 ;                                              /* 0x0000001500097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f80*/                   MOV R10, R22 ;                                             /* 0x00000016000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3f90*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3fa0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3fb0*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3fc0*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3fd0*/                   MOV R11, R7 ;                                              /* 0x00000007000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3fe0*/                   MOV R18, R8 ;                                              /* 0x0000000800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*3ff0*/                   IMAD.U32 R19, R9, R11, RZ ;                                /* 0x0000000b09137224 */
                                                                                              /* 0x003fde00078e00ff */
        /*4000*/                   IMAD.HI.U32 R20, R9, R11, RZ ;                             /* 0x0000000b09147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4010*/                   IMAD.U32 R18, R9, R18, R20 ;                               /* 0x0000001209127224 */
                                                                                              /* 0x003fde00078e0014 */
        /*4020*/                   IMAD.U32 R10, R10, R11, R18 ;                              /* 0x0000000b0a0a7224 */
                                                                                              /* 0x003fde00078e0012 */
        /*4030*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4040*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4050*/                   IADD3 R11, P0, -R19, RZ, RZ ;                              /* 0x000000ff130b7210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*4060*/                   IADD3.X R10, ~R10, RZ, RZ, P0, !PT ;                       /* 0x000000ff0a0a7210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*4070*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4080*/                   MOV R18, R10 ;                                             /* 0x0000000a00127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4090*/                   MOV R9, R21 ;                                              /* 0x0000001500097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40a0*/                   MOV R10, R22 ;                                             /* 0x00000016000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40b0*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40c0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40d0*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40e0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*40f0*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4100*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4110*/                   MOV R19, R21 ;                                             /* 0x0000001500137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4120*/                   MOV R20, R22 ;                                             /* 0x0000001600147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4130*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4140*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4150*/                   IMAD.HI.U32 R21, R9, R11, RZ ;                             /* 0x0000000b09157227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4160*/                   IMAD.U32 R22, R9, R18, R21 ;                               /* 0x0000001209167224 */
                                                                                              /* 0x003fde00078e0015 */
        /*4170*/                   IMAD R23, R9, R18, RZ ;                                    /* 0x0000001209177224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4180*/                   IADD3 RZ, P0, R23, R21, RZ ;                               /* 0x0000001517ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4190*/                   IMAD.HI.U32 R21, R9, R18, RZ ;                             /* 0x0000001209157227 */
                                                                                              /* 0x003fde00078e00ff */
        /*41a0*/                   IADD3.X R21, R21, R19, RZ, P0, !PT ;                       /* 0x0000001315157210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*41b0*/                   IMAD.HI.U32 R9, R9, R18, RZ ;                              /* 0x0000001209097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*41c0*/                   IADD3.X RZ, P0, R9, R19, RZ, P0, !PT ;                     /* 0x0000001309ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*41d0*/                   IMAD.HI.U32 R9, R10, R18, RZ ;                             /* 0x000000120a097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*41e0*/                   IADD3.X R20, R9, R20, RZ, P0, !PT ;                        /* 0x0000001409147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*41f0*/                   IMAD R9, R10, R11, RZ ;                                    /* 0x0000000b0a097224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4200*/                   IADD3 RZ, P0, R9, R22, RZ ;                                /* 0x0000001609ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4210*/                   IMAD.HI.U32 R9, R10, R11, RZ ;                             /* 0x0000000b0a097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4220*/                   IADD3.X R9, R9, R21, RZ, P0, !PT ;                         /* 0x0000001509097210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4230*/                   IMAD.HI.U32 R11, R10, R11, RZ ;                            /* 0x0000000b0a0b7227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4240*/                   IADD3.X RZ, P0, R11, R21, RZ, P0, !PT ;                    /* 0x000000150bff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*4250*/                   IADD3.X R20, R20, RZ, RZ, P0, !PT ;                        /* 0x000000ff14147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4260*/                   IMAD.U32 R11, R10, R18, R9 ;                               /* 0x000000120a0b7224 */
                                                                                              /* 0x003fde00078e0009 */
        /*4270*/                   IMAD R10, R10, R18, RZ ;                                   /* 0x000000120a0a7224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4280*/                   IADD3 RZ, P0, R10, R9, RZ ;                                /* 0x000000090aff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4290*/                   IADD3.X R20, R20, RZ, RZ, P0, !PT ;                        /* 0x000000ff14147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*42a0*/                   MOV R21, R11 ;                                             /* 0x0000000b00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*42b0*/                   MOV R22, R20 ;                                             /* 0x0000001400167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*42c0*/                   MOV R9, R21 ;                                              /* 0x0000001500097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*42d0*/                   MOV R10, R22 ;                                             /* 0x00000016000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*42e0*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*42f0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4300*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4310*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4320*/                   MOV R11, R7 ;                                              /* 0x00000007000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4330*/                   MOV R18, R8 ;                                              /* 0x0000000800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4340*/                   IMAD.U32 R19, R9, R11, RZ ;                                /* 0x0000000b09137224 */
                                                                                              /* 0x003fde00078e00ff */
        /*4350*/                   IMAD.HI.U32 R20, R9, R11, RZ ;                             /* 0x0000000b09147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4360*/                   IMAD.U32 R18, R9, R18, R20 ;                               /* 0x0000001209127224 */
                                                                                              /* 0x003fde00078e0014 */
        /*4370*/                   IMAD.U32 R10, R10, R11, R18 ;                              /* 0x0000000b0a0a7224 */
                                                                                              /* 0x003fde00078e0012 */
        /*4380*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4390*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*43a0*/                   IADD3 R11, P0, -R19, RZ, RZ ;                              /* 0x000000ff130b7210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*43b0*/                   IADD3.X R10, ~R10, RZ, RZ, P0, !PT ;                       /* 0x000000ff0a0a7210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*43c0*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*43d0*/                   MOV R18, R10 ;                                             /* 0x0000000a00127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*43e0*/                   MOV R9, R21 ;                                              /* 0x0000001500097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*43f0*/                   MOV R10, R22 ;                                             /* 0x00000016000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4400*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4410*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4420*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4430*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4440*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4450*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4460*/                   MOV R19, R21 ;                                             /* 0x0000001500137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4470*/                   MOV R20, R22 ;                                             /* 0x0000001600147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4480*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4490*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*44a0*/                   IMAD.HI.U32 R21, R9, R11, RZ ;                             /* 0x0000000b09157227 */
                                                                                              /* 0x003fde00078e00ff */
        /*44b0*/                   IMAD.U32 R22, R9, R18, R21 ;                               /* 0x0000001209167224 */
                                                                                              /* 0x003fde00078e0015 */
        /*44c0*/                   IMAD R23, R9, R18, RZ ;                                    /* 0x0000001209177224 */
                                                                                              /* 0x003fde00078e02ff */
        /*44d0*/                   IADD3 RZ, P0, R23, R21, RZ ;                               /* 0x0000001517ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*44e0*/                   IMAD.HI.U32 R21, R9, R18, RZ ;                             /* 0x0000001209157227 */
                                                                                              /* 0x003fde00078e00ff */
        /*44f0*/                   IADD3.X R21, R21, R19, RZ, P0, !PT ;                       /* 0x0000001315157210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4500*/                   IMAD.HI.U32 R9, R9, R18, RZ ;                              /* 0x0000001209097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4510*/                   IADD3.X RZ, P0, R9, R19, RZ, P0, !PT ;                     /* 0x0000001309ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*4520*/                   IMAD.HI.U32 R9, R10, R18, RZ ;                             /* 0x000000120a097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4530*/                   IADD3.X R20, R9, R20, RZ, P0, !PT ;                        /* 0x0000001409147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4540*/                   IMAD R9, R10, R11, RZ ;                                    /* 0x0000000b0a097224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4550*/                   IADD3 RZ, P0, R9, R22, RZ ;                                /* 0x0000001609ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4560*/                   IMAD.HI.U32 R9, R10, R11, RZ ;                             /* 0x0000000b0a097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4570*/                   IADD3.X R9, R9, R21, RZ, P0, !PT ;                         /* 0x0000001509097210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4580*/                   IMAD.HI.U32 R11, R10, R11, RZ ;                            /* 0x0000000b0a0b7227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4590*/                   IADD3.X RZ, P0, R11, R21, RZ, P0, !PT ;                    /* 0x000000150bff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*45a0*/                   IADD3.X R20, R20, RZ, RZ, P0, !PT ;                        /* 0x000000ff14147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*45b0*/                   IMAD.U32 R11, R10, R18, R9 ;                               /* 0x000000120a0b7224 */
                                                                                              /* 0x003fde00078e0009 */
        /*45c0*/                   IMAD R10, R10, R18, RZ ;                                   /* 0x000000120a0a7224 */
                                                                                              /* 0x003fde00078e02ff */
        /*45d0*/                   IADD3 RZ, P0, R10, R9, RZ ;                                /* 0x000000090aff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*45e0*/                   IADD3.X R20, R20, RZ, RZ, P0, !PT ;                        /* 0x000000ff14147210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*45f0*/                   MOV R9, R11 ;                                              /* 0x0000000b00097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4600*/                   MOV R10, R20 ;                                             /* 0x00000014000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4610*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4620*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4630*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4640*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4650*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4660*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4670*/                   MOV R11, R6 ;                                              /* 0x00000006000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4680*/                   MOV R18, R5 ;                                              /* 0x0000000500127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4690*/                   IMAD.HI.U32 R19, R9, R11, RZ ;                             /* 0x0000000b09137227 */
                                                                                              /* 0x003fde00078e00ff */
        /*46a0*/                   IMAD.U32 R20, R9, R18, R19 ;                               /* 0x0000001209147224 */
                                                                                              /* 0x003fde00078e0013 */
        /*46b0*/                   IMAD R21, R9, R18, RZ ;                                    /* 0x0000001209157224 */
                                                                                              /* 0x003fde00078e02ff */
        /*46c0*/                   IADD3 RZ, P0, R21, R19, RZ ;                               /* 0x0000001315ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*46d0*/                   IMAD.HI.U32 R9, R9, R18, RZ ;                              /* 0x0000001209097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*46e0*/                   IADD3.X R9, R9, RZ, RZ, P0, !PT ;                          /* 0x000000ff09097210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*46f0*/                   IMAD R19, R10, R11, RZ ;                                   /* 0x0000000b0a137224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4700*/                   IADD3 RZ, P0, R19, R20, RZ ;                               /* 0x0000001413ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4710*/                   IMAD.HI.U32 R19, R10, R11, RZ ;                            /* 0x0000000b0a137227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4720*/                   IADD3.X R19, R19, R9, RZ, P0, !PT ;                        /* 0x0000000913137210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4730*/                   IMAD.HI.U32 R11, R10, R11, RZ ;                            /* 0x0000000b0a0b7227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4740*/                   IADD3.X RZ, P0, R11, R9, RZ, P0, !PT ;                     /* 0x000000090bff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*4750*/                   IMAD.HI.U32 R9, R10, R18, RZ ;                             /* 0x000000120a097227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4760*/                   IADD3.X R9, R9, RZ, RZ, P0, !PT ;                          /* 0x000000ff09097210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4770*/                   IMAD.U32 R11, R10, R18, R19 ;                              /* 0x000000120a0b7224 */
                                                                                              /* 0x003fde00078e0013 */
        /*4780*/                   IMAD R10, R10, R18, RZ ;                                   /* 0x000000120a0a7224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4790*/                   IADD3 RZ, P0, R10, R19, RZ ;                               /* 0x000000130aff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*47a0*/                   IADD3.X R9, R9, RZ, RZ, P0, !PT ;                          /* 0x000000ff09097210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*47b0*/                   MOV R11, R11 ;                                             /* 0x0000000b000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*47c0*/                   MOV R10, R9 ;                                              /* 0x00000009000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*47d0*/                   MOV R9, R11 ;                                              /* 0x0000000b00097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*47e0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*47f0*/                   MOV R9, R9 ;                                               /* 0x0000000900097202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4800*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4810*/                   MOV R7, R7 ;                                               /* 0x0000000700077202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4820*/                   MOV R8, R8 ;                                               /* 0x0000000800087202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4830*/                   MOV R11, R7 ;                                              /* 0x00000007000b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4840*/                   MOV R18, R8 ;                                              /* 0x0000000800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4850*/                   IMAD.U32 R19, R9, R11, RZ ;                                /* 0x0000000b09137224 */
                                                                                              /* 0x003fde00078e00ff */
        /*4860*/                   IMAD.HI.U32 R20, R9, R11, RZ ;                             /* 0x0000000b09147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4870*/                   IMAD.U32 R18, R9, R18, R20 ;                               /* 0x0000001209127224 */
                                                                                              /* 0x003fde00078e0014 */
        /*4880*/                   IMAD.U32 R10, R10, R11, R18 ;                              /* 0x0000000b0a0a7224 */
                                                                                              /* 0x003fde00078e0012 */
        /*4890*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*48a0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*48b0*/                   IADD3 R6, P0, R6, -R19, RZ ;                               /* 0x8000001306067210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*48c0*/                   IADD3.X R10, R5, ~R10, RZ, P0, !PT ;                       /* 0x8000000a050a7210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*48d0*/                   MOV R6, R6 ;                                               /* 0x0000000600067202 */
                                                                                              /* 0x003fde0000000f00 */
        /*48e0*/                   MOV R10, R10 ;                                             /* 0x0000000a000a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*48f0*/                   ISETP.GE.U32.AND P0, PT, R6, R7, PT ;                      /* 0x000000070600720c */
                                                                                              /* 0x003fde0003f06070 */
        /*4900*/                   ISETP.GE.U32.AND.EX P0, PT, R10, R8, PT, P0 ;              /* 0x000000080a00720c */
                                                                                              /* 0x003fde0003f06100 */
        /*4910*/                   IADD3 R5, P1, R6, -R7, RZ ;                                /* 0x8000000706057210 */
                                                                                              /* 0x003fde0007f3e0ff */
        /*4920*/                   IADD3.X R9, R10, ~R8, RZ, P1, !PT ;                        /* 0x800000080a097210 */
                                                                                              /* 0x003fde0000ffe4ff */
        /*4930*/                   SEL R5, R5, R6, P0 ;                                       /* 0x0000000605057207 */
                                                                                              /* 0x003fde0000000000 */
        /*4940*/                   SEL R9, R9, R10, P0 ;                                      /* 0x0000000a09097207 */
                                                                                              /* 0x003fde0000000000 */
        /*4950*/                   ISETP.GE.U32.AND P0, PT, R5, R7, PT ;                      /* 0x000000070500720c */
                                                                                              /* 0x003fde0003f06070 */
        /*4960*/                   ISETP.GE.U32.AND.EX P0, PT, R9, R8, PT, P0 ;               /* 0x000000080900720c */
                                                                                              /* 0x003fde0003f06100 */
        /*4970*/                   IADD3 R6, P1, R5, -R7, RZ ;                                /* 0x8000000705067210 */
                                                                                              /* 0x003fde0007f3e0ff */
        /*4980*/                   IADD3.X R7, R9, ~R8, RZ, P1, !PT ;                         /* 0x8000000809077210 */
                                                                                              /* 0x003fde0000ffe4ff */
        /*4990*/                   SEL R6, R6, R5, P0 ;                                       /* 0x0000000506067207 */
                                                                                              /* 0x003fde0000000000 */
        /*49a0*/                   SEL R7, R7, R9, P0 ;                                       /* 0x0000000907077207 */
                                                                                              /* 0x003fde0000000000 */
        /*49b0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*49c0*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*49d0*/                   ISETP.LT.AND P0, PT, R2, RZ, PT ;                          /* 0x000000ff0200720c */
                                                                                              /* 0x003fde0003f01270 */
        /*49e0*/                   IADD3 R2, P1, -R6, RZ, RZ ;                                /* 0x000000ff06027210 */
                                                                                              /* 0x003fde0007f3e1ff */
        /*49f0*/                   IADD3.X R5, ~R7, RZ, RZ, P1, !PT ;                         /* 0x000000ff07057210 */
                                                                                              /* 0x003fde0000ffe5ff */
        /*4a00*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4a10*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4a20*/                   SEL R2, R2, R6, P0 ;                                       /* 0x0000000602027207 */
                                                                                              /* 0x003fde0000000000 */
        /*4a30*/                   SEL R5, R5, R7, P0 ;                                       /* 0x0000000705057207 */
                                                                                              /* 0x003fde0000000000 */
        /*4a40*/                   ISETP.EQ.U32.AND P0, PT, R3, RZ, PT ;                      /* 0x000000ff0300720c */
                                                                                              /* 0x003fde0003f02070 */
        /*4a50*/                   ISETP.EQ.AND.EX P0, PT, R4, RZ, PT, P0 ;                   /* 0x000000ff0400720c */
                                                                                              /* 0x003fde0003f02300 */
        /*4a60*/                   SEL R4, R2, 0xffffffff, !P0 ;                              /* 0xffffffff02047807 */
                                                                                              /* 0x003fde0004000000 */
        /*4a70*/                   SEL R5, R5, 0xffffffff, !P0 ;                              /* 0xffffffff05057807 */
                                                                                              /* 0x003fde0004000000 */
        /*4a80*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4a90*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4aa0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ab0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ac0*/                   MOV R2, R0 ;                                               /* 0x0000000000027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ad0*/                   MOV R3, 0x0 ;                                              /* 0x0000000000037802 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ae0*/                   RET.REL.NODEC R2 0x0 ;                                     /* 0xffffb51002007950 */
                                                                                              /* 0x003fde0003c3ffff */
        /*4af0*/                   MOV R2, R18 ;                                              /* 0x0000001200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b00*/                   MOV R3, R19 ;                                              /* 0x0000001300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b10*/                   MOV R4, R20 ;                                              /* 0x0000001400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b20*/                   MOV R5, R21 ;                                              /* 0x0000001500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b30*/                   MOV R18, R2 ;                                              /* 0x0000000200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b40*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b50*/                   MOV R3, R18 ;                                              /* 0x0000001200037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b60*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b70*/                   MOV R18, R4 ;                                              /* 0x0000000400127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b80*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4b90*/                   MOV R4, R18 ;                                              /* 0x0000001200047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ba0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4bb0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4bc0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4bd0*/                   I2F.U64.RP R18, R4 ;                                       /* 0x0000000400127312 */
                                                                                              /* 0x00321e0000309000 */
        /*4be0*/                   MUFU.RCP R18, R18 ;                                        /* 0x0000001200127308 */
                                                                                              /* 0x00321e0000001000 */
        /*4bf0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c00*/                   IADD3 R18, R18, 0x1ffffffe, RZ ;                           /* 0x1ffffffe12127810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*4c10*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c20*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c30*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c40*/                   F2I.U64.TRUNC R18, R18 ;                                   /* 0x0000001200127311 */
                                                                                              /* 0x00321e000020d800 */
        /*4c50*/                   MOV R24, R18 ;                                             /* 0x0000001200187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c60*/                   MOV R25, R19 ;                                             /* 0x0000001300197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c70*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c80*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4c90*/                   MOV R18, R24 ;                                             /* 0x0000001800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ca0*/                   MOV R19, R25 ;                                             /* 0x0000001900137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4cb0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4cc0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4cd0*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ce0*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4cf0*/                   MOV R20, R4 ;                                              /* 0x0000000400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4d00*/                   MOV R21, R5 ;                                              /* 0x0000000500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4d10*/                   IMAD.U32 R22, R18, R20, RZ ;                               /* 0x0000001412167224 */
                                                                                              /* 0x003fde00078e00ff */
        /*4d20*/                   IMAD.HI.U32 R23, R18, R20, RZ ;                            /* 0x0000001412177227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4d30*/                   IMAD.U32 R21, R18, R21, R23 ;                              /* 0x0000001512157224 */
                                                                                              /* 0x003fde00078e0017 */
        /*4d40*/                   IMAD.U32 R19, R19, R20, R21 ;                              /* 0x0000001413137224 */
                                                                                              /* 0x003fde00078e0015 */
        /*4d50*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4d60*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4d70*/                   IADD3 R20, P0, -R22, RZ, RZ ;                              /* 0x000000ff16147210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*4d80*/                   IADD3.X R19, ~R19, RZ, RZ, P0, !PT ;                       /* 0x000000ff13137210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*4d90*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4da0*/                   MOV R21, R19 ;                                             /* 0x0000001300157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4db0*/                   MOV R18, R24 ;                                             /* 0x0000001800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4dc0*/                   MOV R19, R25 ;                                             /* 0x0000001900137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4dd0*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4de0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4df0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e00*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e10*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e20*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e30*/                   MOV R22, R24 ;                                             /* 0x0000001800167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e40*/                   MOV R23, R25 ;                                             /* 0x0000001900177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e50*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e60*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4e70*/                   IMAD.HI.U32 R24, R18, R20, RZ ;                            /* 0x0000001412187227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4e80*/                   IMAD.U32 R25, R18, R21, R24 ;                              /* 0x0000001512197224 */
                                                                                              /* 0x003fde00078e0018 */
        /*4e90*/                   IMAD R26, R18, R21, RZ ;                                   /* 0x00000015121a7224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4ea0*/                   IADD3 RZ, P0, R26, R24, RZ ;                               /* 0x000000181aff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4eb0*/                   IMAD.HI.U32 R24, R18, R21, RZ ;                            /* 0x0000001512187227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4ec0*/                   IADD3.X R24, R24, R22, RZ, P0, !PT ;                       /* 0x0000001618187210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4ed0*/                   IMAD.HI.U32 R18, R18, R21, RZ ;                            /* 0x0000001512127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4ee0*/                   IADD3.X RZ, P0, R18, R22, RZ, P0, !PT ;                    /* 0x0000001612ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*4ef0*/                   IMAD.HI.U32 R18, R19, R21, RZ ;                            /* 0x0000001513127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4f00*/                   IADD3.X R23, R18, R23, RZ, P0, !PT ;                       /* 0x0000001712177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4f10*/                   IMAD R18, R19, R20, RZ ;                                   /* 0x0000001413127224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4f20*/                   IADD3 RZ, P0, R18, R25, RZ ;                               /* 0x0000001912ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4f30*/                   IMAD.HI.U32 R18, R19, R20, RZ ;                            /* 0x0000001413127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4f40*/                   IADD3.X R18, R18, R24, RZ, P0, !PT ;                       /* 0x0000001812127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4f50*/                   IMAD.HI.U32 R20, R19, R20, RZ ;                            /* 0x0000001413147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*4f60*/                   IADD3.X RZ, P0, R20, R24, RZ, P0, !PT ;                    /* 0x0000001814ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*4f70*/                   IADD3.X R23, R23, RZ, RZ, P0, !PT ;                        /* 0x000000ff17177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4f80*/                   IMAD.U32 R20, R19, R21, R18 ;                              /* 0x0000001513147224 */
                                                                                              /* 0x003fde00078e0012 */
        /*4f90*/                   IMAD R19, R19, R21, RZ ;                                   /* 0x0000001513137224 */
                                                                                              /* 0x003fde00078e02ff */
        /*4fa0*/                   IADD3 RZ, P0, R19, R18, RZ ;                               /* 0x0000001213ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*4fb0*/                   IADD3.X R23, R23, RZ, RZ, P0, !PT ;                        /* 0x000000ff17177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*4fc0*/                   MOV R24, R20 ;                                             /* 0x0000001400187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4fd0*/                   MOV R25, R23 ;                                             /* 0x0000001700197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4fe0*/                   MOV R18, R24 ;                                             /* 0x0000001800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*4ff0*/                   MOV R19, R25 ;                                             /* 0x0000001900137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5000*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5010*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5020*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5030*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5040*/                   MOV R20, R4 ;                                              /* 0x0000000400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5050*/                   MOV R21, R5 ;                                              /* 0x0000000500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5060*/                   IMAD.U32 R22, R18, R20, RZ ;                               /* 0x0000001412167224 */
                                                                                              /* 0x003fde00078e00ff */
        /*5070*/                   IMAD.HI.U32 R23, R18, R20, RZ ;                            /* 0x0000001412177227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5080*/                   IMAD.U32 R21, R18, R21, R23 ;                              /* 0x0000001512157224 */
                                                                                              /* 0x003fde00078e0017 */
        /*5090*/                   IMAD.U32 R19, R19, R20, R21 ;                              /* 0x0000001413137224 */
                                                                                              /* 0x003fde00078e0015 */
        /*50a0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*50b0*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*50c0*/                   IADD3 R20, P0, -R22, RZ, RZ ;                              /* 0x000000ff16147210 */
                                                                                              /* 0x003fde0007f1e1ff */
        /*50d0*/                   IADD3.X R19, ~R19, RZ, RZ, P0, !PT ;                       /* 0x000000ff13137210 */
                                                                                              /* 0x003fde00007fe5ff */
        /*50e0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*50f0*/                   MOV R21, R19 ;                                             /* 0x0000001300157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5100*/                   MOV R18, R24 ;                                             /* 0x0000001800127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5110*/                   MOV R19, R25 ;                                             /* 0x0000001900137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5120*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5130*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5140*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5150*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5160*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5170*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5180*/                   MOV R22, R24 ;                                             /* 0x0000001800167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5190*/                   MOV R23, R25 ;                                             /* 0x0000001900177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*51a0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*51b0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*51c0*/                   IMAD.HI.U32 R24, R18, R20, RZ ;                            /* 0x0000001412187227 */
                                                                                              /* 0x003fde00078e00ff */
        /*51d0*/                   IMAD.U32 R25, R18, R21, R24 ;                              /* 0x0000001512197224 */
                                                                                              /* 0x003fde00078e0018 */
        /*51e0*/                   IMAD R26, R18, R21, RZ ;                                   /* 0x00000015121a7224 */
                                                                                              /* 0x003fde00078e02ff */
        /*51f0*/                   IADD3 RZ, P0, R26, R24, RZ ;                               /* 0x000000181aff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*5200*/                   IMAD.HI.U32 R24, R18, R21, RZ ;                            /* 0x0000001512187227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5210*/                   IADD3.X R24, R24, R22, RZ, P0, !PT ;                       /* 0x0000001618187210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5220*/                   IMAD.HI.U32 R18, R18, R21, RZ ;                            /* 0x0000001512127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5230*/                   IADD3.X RZ, P0, R18, R22, RZ, P0, !PT ;                    /* 0x0000001612ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*5240*/                   IMAD.HI.U32 R18, R19, R21, RZ ;                            /* 0x0000001513127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5250*/                   IADD3.X R23, R18, R23, RZ, P0, !PT ;                       /* 0x0000001712177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5260*/                   IMAD R18, R19, R20, RZ ;                                   /* 0x0000001413127224 */
                                                                                              /* 0x003fde00078e02ff */
        /*5270*/                   IADD3 RZ, P0, R18, R25, RZ ;                               /* 0x0000001912ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*5280*/                   IMAD.HI.U32 R18, R19, R20, RZ ;                            /* 0x0000001413127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5290*/                   IADD3.X R18, R18, R24, RZ, P0, !PT ;                       /* 0x0000001812127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*52a0*/                   IMAD.HI.U32 R20, R19, R20, RZ ;                            /* 0x0000001413147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*52b0*/                   IADD3.X RZ, P0, R20, R24, RZ, P0, !PT ;                    /* 0x0000001814ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*52c0*/                   IADD3.X R23, R23, RZ, RZ, P0, !PT ;                        /* 0x000000ff17177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*52d0*/                   IMAD.U32 R20, R19, R21, R18 ;                              /* 0x0000001513147224 */
                                                                                              /* 0x003fde00078e0012 */
        /*52e0*/                   IMAD R19, R19, R21, RZ ;                                   /* 0x0000001513137224 */
                                                                                              /* 0x003fde00078e02ff */
        /*52f0*/                   IADD3 RZ, P0, R19, R18, RZ ;                               /* 0x0000001213ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*5300*/                   IADD3.X R23, R23, RZ, RZ, P0, !PT ;                        /* 0x000000ff17177210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5310*/                   MOV R18, R20 ;                                             /* 0x0000001400127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5320*/                   MOV R19, R23 ;                                             /* 0x0000001700137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5330*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5340*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5350*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5360*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5370*/                   MOV R3, R3 ;                                               /* 0x0000000300037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5380*/                   MOV R2, R2 ;                                               /* 0x0000000200027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5390*/                   MOV R20, R3 ;                                              /* 0x0000000300147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*53a0*/                   MOV R21, R2 ;                                              /* 0x0000000200157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*53b0*/                   IMAD.HI.U32 R22, R18, R20, RZ ;                            /* 0x0000001412167227 */
                                                                                              /* 0x003fde00078e00ff */
        /*53c0*/                   IMAD.U32 R23, R18, R21, R22 ;                              /* 0x0000001512177224 */
                                                                                              /* 0x003fde00078e0016 */
        /*53d0*/                   IMAD R24, R18, R21, RZ ;                                   /* 0x0000001512187224 */
                                                                                              /* 0x003fde00078e02ff */
        /*53e0*/                   IADD3 RZ, P0, R24, R22, RZ ;                               /* 0x0000001618ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*53f0*/                   IMAD.HI.U32 R18, R18, R21, RZ ;                            /* 0x0000001512127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5400*/                   IADD3.X R18, R18, RZ, RZ, P0, !PT ;                        /* 0x000000ff12127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5410*/                   IMAD R22, R19, R20, RZ ;                                   /* 0x0000001413167224 */
                                                                                              /* 0x003fde00078e02ff */
        /*5420*/                   IADD3 RZ, P0, R22, R23, RZ ;                               /* 0x0000001716ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*5430*/                   IMAD.HI.U32 R22, R19, R20, RZ ;                            /* 0x0000001413167227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5440*/                   IADD3.X R22, R22, R18, RZ, P0, !PT ;                       /* 0x0000001216167210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5450*/                   IMAD.HI.U32 R20, R19, R20, RZ ;                            /* 0x0000001413147227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5460*/                   IADD3.X RZ, P0, R20, R18, RZ, P0, !PT ;                    /* 0x0000001214ff7210 */
                                                                                              /* 0x003fde000071e4ff */
        /*5470*/                   IMAD.HI.U32 R18, R19, R21, RZ ;                            /* 0x0000001513127227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5480*/                   IADD3.X R18, R18, RZ, RZ, P0, !PT ;                        /* 0x000000ff12127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*5490*/                   IMAD.U32 R20, R19, R21, R22 ;                              /* 0x0000001513147224 */
                                                                                              /* 0x003fde00078e0016 */
        /*54a0*/                   IMAD R19, R19, R21, RZ ;                                   /* 0x0000001513137224 */
                                                                                              /* 0x003fde00078e02ff */
        /*54b0*/                   IADD3 RZ, P0, R19, R22, RZ ;                               /* 0x0000001613ff7210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*54c0*/                   IADD3.X R18, R18, RZ, RZ, P0, !PT ;                        /* 0x000000ff12127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*54d0*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*54e0*/                   MOV R19, R18 ;                                             /* 0x0000001200137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*54f0*/                   MOV R18, R20 ;                                             /* 0x0000001400127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5500*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5510*/                   MOV R18, R18 ;                                             /* 0x0000001200127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5520*/                   MOV R19, R19 ;                                             /* 0x0000001300137202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5530*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5540*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5550*/                   MOV R20, R4 ;                                              /* 0x0000000400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5560*/                   MOV R21, R5 ;                                              /* 0x0000000500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5570*/                   IMAD.U32 R22, R18, R20, RZ ;                               /* 0x0000001412167224 */
                                                                                              /* 0x003fde00078e00ff */
        /*5580*/                   IMAD.HI.U32 R23, R18, R20, RZ ;                            /* 0x0000001412177227 */
                                                                                              /* 0x003fde00078e00ff */
        /*5590*/                   IMAD.U32 R21, R18, R21, R23 ;                              /* 0x0000001512157224 */
                                                                                              /* 0x003fde00078e0017 */
        /*55a0*/                   IMAD.U32 R19, R19, R20, R21 ;                              /* 0x0000001413137224 */
                                                                                              /* 0x003fde00078e0015 */
        /*55b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*55c0*/                   MOV R18, R19 ;                                             /* 0x0000001300127202 */
                                                                                              /* 0x003fde0000000f00 */
        /*55d0*/                   IADD3 R3, P0, R3, -R22, RZ ;                               /* 0x8000001603037210 */
                                                                                              /* 0x003fde0007f1e0ff */
        /*55e0*/                   IADD3.X R18, R2, ~R18, RZ, P0, !PT ;                       /* 0x8000001202127210 */
                                                                                              /* 0x003fde00007fe4ff */
        /*55f0*/                   MOV R2, R3 ;                                               /* 0x0000000300027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5600*/                   MOV R3, R18 ;                                              /* 0x0000001200037202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5610*/                   ISETP.GE.U32.AND P0, PT, R2, R4, PT ;                      /* 0x000000040200720c */
                                                                                              /* 0x003fde0003f06070 */
        /*5620*/                   ISETP.GE.U32.AND.EX P0, PT, R3, R5, PT, P0 ;               /* 0x000000050300720c */
                                                                                              /* 0x003fde0003f06100 */
        /*5630*/                   IADD3 R18, P1, R2, -R4, RZ ;                               /* 0x8000000402127210 */
                                                                                              /* 0x003fde0007f3e0ff */
        /*5640*/                   IADD3.X R19, R3, ~R5, RZ, P1, !PT ;                        /* 0x8000000503137210 */
                                                                                              /* 0x003fde0000ffe4ff */
        /*5650*/                   SEL R18, R18, R2, P0 ;                                     /* 0x0000000212127207 */
                                                                                              /* 0x003fde0000000000 */
        /*5660*/                   SEL R19, R19, R3, P0 ;                                     /* 0x0000000313137207 */
                                                                                              /* 0x003fde0000000000 */
        /*5670*/                   ISETP.GE.U32.AND P0, PT, R18, R4, PT ;                     /* 0x000000041200720c */
                                                                                              /* 0x003fde0003f06070 */
        /*5680*/                   ISETP.GE.U32.AND.EX P0, PT, R19, R5, PT, P0 ;              /* 0x000000051300720c */
                                                                                              /* 0x003fde0003f06100 */
        /*5690*/                   IADD3 R2, P1, R18, -R4, RZ ;                               /* 0x8000000412027210 */
                                                                                              /* 0x003fde0007f3e0ff */
        /*56a0*/                   IADD3.X R3, R19, ~R5, RZ, P1, !PT ;                        /* 0x8000000513037210 */
                                                                                              /* 0x003fde0000ffe4ff */
        /*56b0*/                   SEL R2, R2, R18, P0 ;                                      /* 0x0000001202027207 */
                                                                                              /* 0x003fde0000000000 */
        /*56c0*/                   SEL R3, R3, R19, P0 ;                                      /* 0x0000001303037207 */
                                                                                              /* 0x003fde0000000000 */
        /*56d0*/                   ISETP.EQ.U32.AND P0, PT, R4, RZ, PT ;                      /* 0x000000ff0400720c */
                                                                                              /* 0x003fde0003f02070 */
        /*56e0*/                   ISETP.EQ.AND.EX P0, PT, R5, RZ, PT, P0 ;                   /* 0x000000ff0500720c */
                                                                                              /* 0x003fde0003f02300 */
        /*56f0*/                   SEL R4, R2, 0xffffffff, !P0 ;                              /* 0xffffffff02047807 */
                                                                                              /* 0x003fde0004000000 */
        /*5700*/                   SEL R5, R3, 0xffffffff, !P0 ;                              /* 0xffffffff03057807 */
                                                                                              /* 0x003fde0004000000 */
        /*5710*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5720*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5730*/                   MOV R4, R4 ;                                               /* 0x0000000400047202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5740*/                   MOV R5, R5 ;                                               /* 0x0000000500057202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5750*/                   MOV R2, R0 ;                                               /* 0x0000000000027202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5760*/                   MOV R3, 0x0 ;                                              /* 0x0000000000037802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5770*/                   RET.REL.NODEC R2 0x0 ;                                     /* 0xffffa88002007950 */
                                                                                              /* 0x003fde0003c3ffff */
        /*5780*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5790*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*57a0*/                   MOV R21, R25 ;                                             /* 0x0000001900157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*57b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*57c0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*57d0*/                   SHF.R.U32.HI R23, RZ, 0x17, R21 ;                          /* 0x00000017ff177819 */
                                                                                              /* 0x003fde0000011615 */
        /*57e0*/                   LOP3.LUT R23, R23, 0xff, RZ, 0xc0, !PT ;                   /* 0x000000ff17177812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*57f0*/                   IADD3 R24, R23, -0x1, RZ ;                                 /* 0xffffffff17187810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5800*/                   MOV R26, R22 ;                                             /* 0x00000016001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5810*/                   SHF.R.U32.HI R27, RZ, 0x17, R26 ;                          /* 0x00000017ff1b7819 */
                                                                                              /* 0x003fde000001161a */
        /*5820*/                   LOP3.LUT R27, R27, 0xff, RZ, 0xc0, !PT ;                   /* 0x000000ff1b1b7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5830*/                   IADD3 R28, R27, -0x1, RZ ;                                 /* 0xffffffff1b1c7810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5840*/                   ISETP.GT.U32.AND P0, PT, R24, 0xfd, PT ;                   /* 0x000000fd1800780c */
                                                                                              /* 0x003fde0003f04070 */
        /*5850*/                   ISETP.GT.U32.AND P1, PT, R28, 0xfd, PT ;                   /* 0x000000fd1c00780c */
                                                                                              /* 0x003fde0003f24070 */
        /*5860*/                   PLOP3.LUT P0, PT, P0, P1, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000703570 */
        /*5870*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5880*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5890*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58a0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58b0*/                   MOV R29, R24 ;                                             /* 0x00000018001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58c0*/                   MOV R24, R26 ;                                             /* 0x0000001a00187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58d0*/                   MOV R26, R27 ;                                             /* 0x0000001b001a7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58e0*/                   MOV R27, R28 ;                                             /* 0x0000001c001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*58f0*/               @P0 BRA 0x5930 ;                                               /* 0x0000003000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5900*/                   MOV R25, RZ ;                                              /* 0x000000ff00197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5910*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5920*/                   BRA 0x5ca0 ;                                               /* 0x0000037000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*5930*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5940*/                   FADD.FTZ R28, -RZ, |R25| ;                                 /* 0x40000019ff1c7221 */
                                                                                              /* 0x003fde0000010100 */
        /*5950*/                   MOV R30, 0x7f800000 ;                                      /* 0x7f800000001e7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5960*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5970*/                   FSETP.GTU.FTZ.AND P0, PT, R28, R30, PT ;                   /* 0x0000001e1c00720b */
                                                                                              /* 0x003fde0003f1c000 */
        /*5980*/                   MOV R28, R28 ;                                             /* 0x0000001c001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5990*/                   MOV R30, R30 ;                                             /* 0x0000001e001e7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*59a0*/               @P0 BRA 0x6310 ;                                               /* 0x0000096000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*59b0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*59c0*/                   FADD.FTZ R31, -RZ, |R22| ;                                 /* 0x40000016ff1f7221 */
                                                                                              /* 0x003fde0000010100 */
        /*59d0*/                   FSETP.GTU.FTZ.AND P0, PT, R31, R30, PT ;                   /* 0x0000001e1f00720b */
                                                                                              /* 0x003fde0003f1c000 */
        /*59e0*/                   MOV R31, R31 ;                                             /* 0x0000001f001f7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*59f0*/               @P0 BRA 0x6310 ;                                               /* 0x0000091000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5a00*/                   LOP3.LUT R32, R24, R21, RZ, 0xfc, !PT ;                    /* 0x0000001518207212 */
                                                                                              /* 0x003fde00078efcff */
        /*5a10*/                   LOP3.LUT R32, R32, 0x7fffffff, RZ, 0xc0, !PT ;             /* 0x7fffffff20207812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5a20*/                   ISETP.EQ.AND P0, PT, R32, RZ, PT ;                         /* 0x000000ff2000720c */
                                                                                              /* 0x003fde0003f02270 */
        /*5a30*/               @P0 BRA 0x62b0 ;                                               /* 0x0000087000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5a40*/                   FSETP.EQ.FTZ.AND P0, PT, R28, R30, PT ;                    /* 0x0000001e1c00720b */
                                                                                              /* 0x003fde0003f12000 */
        /*5a50*/                   FSETP.EQ.FTZ.AND P1, PT, R31, R30, PT ;                    /* 0x0000001e1f00720b */
                                                                                              /* 0x003fde0003f32000 */
        /*5a60*/                   PLOP3.LUT P2, PT, P0, P1, PT, 0x80, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000743070 */
        /*5a70*/                   PLOP3.LUT P0, PT, P0, PT, PT, 0x80, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde000070f070 */
        /*5a80*/                   PLOP3.LUT P1, PT, P1, PT, PT, 0x80, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000f2f070 */
        /*5a90*/               @P2 BRA 0x62b0 ;                                               /* 0x0000081000002947 */
                                                                                              /* 0x003fde0003800000 */
        /*5aa0*/                   LOP3.LUT R28, R21, 0x7fffffff, RZ, 0xc0, !PT ;             /* 0x7fffffff151c7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5ab0*/                   ISETP.EQ.AND P2, PT, R28, RZ, PT ;                         /* 0x000000ff1c00720c */
                                                                                              /* 0x003fde0003f42270 */
        /*5ac0*/                   PLOP3.LUT P1, PT, P1, P2, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000f25570 */
        /*5ad0*/               @P1 BRA 0x6250 ;                                               /* 0x0000077000001947 */
                                                                                              /* 0x003fde0003800000 */
        /*5ae0*/                   LOP3.LUT R28, R24, 0x7fffffff, RZ, 0xc0, !PT ;             /* 0x7fffffff181c7812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5af0*/                   ISETP.EQ.AND P1, PT, R28, RZ, PT ;                         /* 0x000000ff1c00720c */
                                                                                              /* 0x003fde0003f22270 */
        /*5b00*/                   PLOP3.LUT P0, PT, P0, P1, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000703570 */
        /*5b10*/               @P0 BRA 0x61e0 ;                                               /* 0x000006c000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5b20*/                   ISETP.LT.AND P0, PT, R29, RZ, PT ;                         /* 0x000000ff1d00720c */
                                                                                              /* 0x003fde0003f01270 */
        /*5b30*/               @P0 BRA 0x5b70 ;                                               /* 0x0000003000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5b40*/                   MOV R25, RZ ;                                              /* 0x000000ff00197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5b50*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5b60*/                   BRA 0x5bf0 ;                                               /* 0x0000008000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*5b70*/                   MOV R21, RZ ;                                              /* 0x000000ff00157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5b80*/                   MOV R28, 0x5f800000 ;                                      /* 0x5f800000001c7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5b90*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ba0*/                   FFMA R21, R25, R28, R21 ;                                  /* 0x0000001c19157223 */
                                                                                              /* 0x003fde0000000015 */
        /*5bb0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5bc0*/                   MOV R25, 0xffffffc0 ;                                      /* 0xffffffc000197802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5bd0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5be0*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5bf0*/                   ISETP.LT.AND P0, PT, R27, RZ, PT ;                         /* 0x000000ff1b00720c */
                                                                                              /* 0x003fde0003f01270 */
        /*5c00*/               @P0 BRA 0x5c20 ;                                               /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5c10*/                   BRA 0x5ca0 ;                                               /* 0x0000008000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*5c20*/                   MOV R24, RZ ;                                              /* 0x000000ff00187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5c30*/                   MOV R27, 0x5f800000 ;                                      /* 0x5f800000001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5c40*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5c50*/                   FFMA R22, R22, R27, R24 ;                                  /* 0x0000001b16167223 */
                                                                                              /* 0x003fde0000000018 */
        /*5c60*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5c70*/                   IADD3 R25, R25, 0x40, RZ ;                                 /* 0x0000004019197810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5c80*/                   MOV R24, R22 ;                                             /* 0x0000001600187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5c90*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ca0*/                   IADD3 R23, R23, -0x7f, RZ ;                                /* 0xffffff8117177810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5cb0*/                   SHF.L.U32 R22, R23, 0x17, RZ ;                             /* 0x0000001717167819 */
                                                                                              /* 0x003fde00000006ff */
        /*5cc0*/                   IADD3 R22, R21, -R22, RZ ;                                 /* 0x8000001615167210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5cd0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ce0*/                   SHF.L.U32 R21, R26, 0x17, RZ ;                             /* 0x000000171a157819 */
                                                                                              /* 0x003fde00000006ff */
        /*5cf0*/                   IADD3 R21, R21, -0x3f800000, RZ ;                          /* 0xc080000015157810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5d00*/                   IADD3 R21, R24, -R21, RZ ;                                 /* 0x8000001518157210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5d10*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5d20*/                   MUFU.RCP R24, R21 ;                                        /* 0x0000001500187308 */
                                                                                              /* 0x00321e0000001000 */
        /*5d30*/                   FADD.FTZ R21, -RZ, -R21 ;                                  /* 0x80000015ff157221 */
                                                                                              /* 0x003fde0000010100 */
        /*5d40*/                   MOV R27, 0x3f800000 ;                                      /* 0x3f800000001b7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5d50*/                   FFMA R27, R21, R24, R27 ;                                  /* 0x00000018151b7223 */
                                                                                              /* 0x003fde000000001b */
        /*5d60*/                   FFMA R27, R24, R27, R24 ;                                  /* 0x0000001b181b7223 */
                                                                                              /* 0x003fde0000000018 */
        /*5d70*/                   MOV R24, RZ ;                                              /* 0x000000ff00187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5d80*/                   FFMA R24, R22, R27, R24 ;                                  /* 0x0000001b16187223 */
                                                                                              /* 0x003fde0000000018 */
        /*5d90*/                   FFMA R28, R21, R24, R22 ;                                  /* 0x00000018151c7223 */
                                                                                              /* 0x003fde0000000016 */
        /*5da0*/                   FFMA R28, R28, R27, R24 ;                                  /* 0x0000001b1c1c7223 */
                                                                                              /* 0x003fde0000000018 */
        /*5db0*/                   FFMA R21, R21, R28, R22 ;                                  /* 0x0000001c15157223 */
                                                                                              /* 0x003fde0000000016 */
        /*5dc0*/                   FFMA R22, R21, R27, R28 ;                                  /* 0x0000001b15167223 */
                                                                                              /* 0x003fde000000001c */
        /*5dd0*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5de0*/                   SHF.R.U32.HI R24, RZ, 0x17, R22 ;                          /* 0x00000017ff187819 */
                                                                                              /* 0x003fde0000011616 */
        /*5df0*/                   LOP3.LUT R24, R24, 0xff, RZ, 0xc0, !PT ;                   /* 0x000000ff18187812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5e00*/                   MOV R29, 0x7f ;                                            /* 0x0000007f001d7802 */
                                                                                              /* 0x003fde0000000f00 */
        /*5e10*/                   IADD3 R29, R29, -R26, RZ ;                                 /* 0x8000001a1d1d7210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5e20*/                   IADD3 R29, R29, R23, RZ ;                                  /* 0x000000171d1d7210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5e30*/                   IADD3 R29, R29, R25, RZ ;                                  /* 0x000000191d1d7210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5e40*/                   IADD3 R24, R29, R24, RZ ;                                  /* 0x000000181d187210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5e50*/                   IADD3 R23, R24, -0x1, RZ ;                                 /* 0xffffffff18177810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5e60*/                   ISETP.LT.U32.AND P0, PT, R23, 0xfe, PT ;                   /* 0x000000fe1700780c */
                                                                                              /* 0x003fde0003f01070 */
        /*5e70*/                   MOV R27, R27 ;                                             /* 0x0000001b001b7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5e80*/                   MOV R28, R28 ;                                             /* 0x0000001c001c7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5e90*/                   MOV R23, R21 ;                                             /* 0x0000001500177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ea0*/                   MOV R21, R22 ;                                             /* 0x0000001600157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5eb0*/                   MOV R29, R29 ;                                             /* 0x0000001d001d7202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ec0*/                   MOV R24, R24 ;                                             /* 0x0000001800187202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5ed0*/               @P0 BRA 0x6170 ;                                               /* 0x0000029000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5ee0*/                   ISETP.GT.AND P0, PT, R24, 0xfe, PT ;                       /* 0x000000fe1800780c */
                                                                                              /* 0x003fde0003f04270 */
        /*5ef0*/               @P0 BRA 0x6130 ;                                               /* 0x0000023000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5f00*/                   ISETP.LT.AND P0, PT, R24, 0x1, PT ;                        /* 0x000000011800780c */
                                                                                              /* 0x003fde0003f01270 */
        /*5f10*/               @P0 BRA 0x5f30 ;                                               /* 0x0000001000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5f20*/                   BRA 0x61a0 ;                                               /* 0x0000027000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*5f30*/                   ISETP.LT.AND P0, PT, R24, -0x18, PT ;                      /* 0xffffffe81800780c */
                                                                                              /* 0x003fde0003f01270 */
        /*5f40*/                   LOP3.LUT R21, R21, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000015157812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5f50*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5f60*/               @P0 BRA 0x61a0 ;                                               /* 0x0000023000000947 */
                                                                                              /* 0x003fde0003800000 */
        /*5f70*/                   FFMA.RP R22, R23, R27, R28 ;                               /* 0x0000001b17167223 */
                                                                                              /* 0x003fde000000801c */
        /*5f80*/                   FFMA.RM R25, R23, R27, R28 ;                               /* 0x0000001b17197223 */
                                                                                              /* 0x003fde000000401c */
        /*5f90*/                   FSETP.NEU.FTZ.AND P0, PT, R22, R25, PT ;                   /* 0x000000191600720b */
                                                                                              /* 0x003fde0003f1d000 */
        /*5fa0*/                   FFMA.RZ R23, R23, R27, R28 ;                               /* 0x0000001b17177223 */
                                                                                              /* 0x003fde000000c01c */
        /*5fb0*/                   MOV R23, R23 ;                                             /* 0x0000001700177202 */
                                                                                              /* 0x003fde0000000f00 */
        /*5fc0*/                   LOP3.LUT R23, R23, 0x7fffff, RZ, 0xc0, !PT ;               /* 0x007fffff17177812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*5fd0*/                   LOP3.LUT R23, R23, 0x800000, RZ, 0xfc, !PT ;               /* 0x0080000017177812 */
                                                                                              /* 0x003fde00078efcff */
        /*5fe0*/                   IADD3 R22, R24, 0x20, RZ ;                                 /* 0x0000002018167810 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*5ff0*/                   SHF.L.U32 R22, R23, R22, RZ ;                              /* 0x0000001617167219 */
                                                                                              /* 0x003fde00000006ff */
        /*6000*/                   ISETP.NE.AND P1, PT, R22, RZ, PT ;                         /* 0x000000ff1600720c */
                                                                                              /* 0x003fde0003f25270 */
        /*6010*/                   ISETP.EQ.AND P2, PT, R24, RZ, PT ;                         /* 0x000000ff1800720c */
                                                                                              /* 0x003fde0003f42270 */
        /*6020*/                   IADD3 R22, RZ, -R24, RZ ;                                  /* 0x80000018ff167210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*6030*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6040*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6050*/                   SEL R22, RZ, R22, P2 ;                                     /* 0x00000016ff167207 */
                                                                                              /* 0x003fde0001000000 */
        /*6060*/                   SHF.R.U32.HI R22, RZ, R22, R23 ;                           /* 0x00000016ff167219 */
                                                                                              /* 0x003fde0000011617 */
        /*6070*/                   ISETP.NE.AND P2, PT, R24, RZ, PT ;                         /* 0x000000ff1800720c */
                                                                                              /* 0x003fde0003f45270 */
        /*6080*/                   PLOP3.LUT P1, PT, P1, P2, PT, 0x80, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000f25070 */
        /*6090*/                   PLOP3.LUT P0, PT, P0, P1, PT, 0xa8, 0x0 ;                  /* 0x000000000000781c */
                                                                                              /* 0x003fde0000703570 */
        /*60a0*/                   SEL R23, RZ, 0x1, !P0 ;                                    /* 0x00000001ff177807 */
                                                                                              /* 0x003fde0004000000 */
        /*60b0*/                   SHF.R.U32.HI R24, RZ, 0x1, R22 ;                           /* 0x00000001ff187819 */
                                                                                              /* 0x003fde0000011616 */
        /*60c0*/                   LOP3.LUT R25, R24, 0x1, RZ, 0xc0, !PT ;                    /* 0x0000000118197812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*60d0*/                   LOP3.LUT R23, R23, R25, RZ, 0xfc, !PT ;                    /* 0x0000001917177212 */
                                                                                              /* 0x003fde00078efcff */
        /*60e0*/                   LOP3.LUT R23, R23, R22, RZ, 0xc0, !PT ;                    /* 0x0000001617177212 */
                                                                                              /* 0x003fde00078ec0ff */
        /*60f0*/                   IADD3 R23, R23, R24, RZ ;                                  /* 0x0000001817177210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*6100*/                   LOP3.LUT R21, R23, R21, RZ, 0xfc, !PT ;                    /* 0x0000001517157212 */
                                                                                              /* 0x003fde00078efcff */
        /*6110*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6120*/                   BRA 0x61a0 ;                                               /* 0x0000007000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*6130*/                   LOP3.LUT R21, R21, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000015157812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*6140*/                   LOP3.LUT R21, R21, 0x7f800000, RZ, 0xfc, !PT ;             /* 0x7f80000015157812 */
                                                                                              /* 0x003fde00078efcff */
        /*6150*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6160*/                   BRA 0x61a0 ;                                               /* 0x0000003000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*6170*/                   SHF.L.U32 R29, R29, 0x17, RZ ;                             /* 0x000000171d1d7819 */
                                                                                              /* 0x003fde00000006ff */
        /*6180*/                   IADD3 R21, R29, R21, RZ ;                                  /* 0x000000151d157210 */
                                                                                              /* 0x003fde0007ffe0ff */
        /*6190*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*61a0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*61b0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*61c0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*61d0*/                   BRA 0x6360 ;                                               /* 0x0000018000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*61e0*/                   LOP3.LUT R21, R24, R21, RZ, 0x3c, !PT ;                    /* 0x0000001518157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*61f0*/                   LOP3.LUT R21, R21, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000015157812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*6200*/                   LOP3.LUT R21, R21, 0x7f800000, RZ, 0xfc, !PT ;             /* 0x7f80000015157812 */
                                                                                              /* 0x003fde00078efcff */
        /*6210*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6220*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6230*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6240*/                   BRA 0x6360 ;                                               /* 0x0000011000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*6250*/                   LOP3.LUT R21, R24, R21, RZ, 0x3c, !PT ;                    /* 0x0000001518157212 */
                                                                                              /* 0x003fde00078e3cff */
        /*6260*/                   LOP3.LUT R21, R21, 0x80000000, RZ, 0xc0, !PT ;             /* 0x8000000015157812 */
                                                                                              /* 0x003fde00078ec0ff */
        /*6270*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6280*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6290*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*62a0*/                   BRA 0x6360 ;                                               /* 0x000000b000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*62b0*/                   MOV R21, 0xffc00000 ;                                      /* 0xffc0000000157802 */
                                                                                              /* 0x003fde0000000f00 */
        /*62c0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*62d0*/                   MUFU.RSQ R21, R21 ;                                        /* 0x0000001500157308 */
                                                                                              /* 0x00321e0000001400 */
        /*62e0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*62f0*/                   MOV R21, R21 ;                                             /* 0x0000001500157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6300*/                   BRA 0x6360 ;                                               /* 0x0000005000007947 */
                                                                                              /* 0x003fde0003800000 */
        /*6310*/                   MOV R25, R25 ;                                             /* 0x0000001900197202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6320*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6330*/                   FADD.FTZ R22, R25, R22 ;                                   /* 0x0000001619167221 */
                                                                                              /* 0x003fde0000010000 */
        /*6340*/                   MOV R22, R22 ;                                             /* 0x0000001600167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6350*/                   MOV R21, R22 ;                                             /* 0x0000001600157202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6360*/                   MOV R22, R21 ;                                             /* 0x0000001500167202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6370*/                   MOV R20, R20 ;                                             /* 0x0000001400147202 */
                                                                                              /* 0x003fde0000000f00 */
        /*6380*/                   MOV R21, 0x0 ;                                             /* 0x0000000000157802 */
                                                                                              /* 0x003fde0000000f00 */
        /*6390*/                   RET.REL.NODEC R20 0x0 ;                                    /* 0xffff9c6014007950 */
                                                                                              /* 0x003fde0003c3ffff */
        /*63a0*/                   BRA 0x63a0;                                                /* 0xfffffff000007947 */
                                                                                              /* 0x000fc0000383ffff */
        /*63b0*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*63c0*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*63d0*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*63e0*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*63f0*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6400*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6410*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6420*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6430*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6440*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6450*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6460*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
        /*6470*/                   NOP;                                                       /* 0x0000000000007918 */
                                                                                              /* 0x000fc00000000000 */
		..........


"""



