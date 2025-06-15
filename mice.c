// gcc mice.c -o mice.exe -O0 -g -mwindows -m64 -nostdlib -Wl,-e_start && ./mice.exe ; echo $?
//  cl mice.c -nologo -Oi -Zi -GS- -Gs9999999 -link -subsystem:windows -nodefaultlib -stack:0x100000,0x100000 -machine:x64 -entry:"_start" && ./mice.exe ; echo $?
// tcc mice.c -o mice.exe -O0 -g -mwindows -m64 -nostdlib && ./mice.exe ; echo $?
// xxd -C mice.exe
// objdump -d mice.exe

typedef          char      i8;
typedef unsigned char      u8;
typedef          short     i16;
typedef unsigned short     u16;
typedef          int       i32;
typedef unsigned int       u32;
typedef          long long i64;
typedef unsigned long long u64;
typedef          float     f32;
typedef          double    f64;
static inline u32 parse_based_int(char* s, u32 i, u64* num, u32 base) {
	for(*num = 0; ; i += 1) {
		     if(s[i] >= '0' && s[i] <= '9') { *num *= base; *num += s[i] - '0'; }
		else if(s[i] >= 'a' && s[i] <= 'f') { *num *= base; *num += s[i] - 'a' + 10; }
		else if(s[i] >= 'A' && s[i] <= 'F') { *num *= base; *num += s[i] - 'A' + 10; }
		else if(s[i] != '_') break;
	}
	return i;
}

#pragma function(memset)
static inline void* memset(void* dest, int val, i64 count) {
	u8* bytes = (u8*)dest;
	while(count--) *bytes++ = (u8)val;
	return dest;
}
#pragma function(memcpy)
static inline void* memcpy(void* dest, void* src, i64 n) {
	u8* destBytes = (u8*)dest;
	u8* srcBytes = (u8*)src;
	while(n--) *destBytes++ = *srcBytes++;
	return dest;
}
#pragma function(strcmp)
static inline int strcmp(char* s1, char* s2) {
	while(*s1 && *s2 && *s1 == *s2) {
		s1++;
		s2++;
	}
	return (*s1 == *s2) ? 0 : *s1 - *s2;
}
static inline int isdigit(int c) {
	return c >= '0' && c <= '9';
}

#ifndef _MSC_VER
static inline u64 __readgsqword(u32 Offset) {
	void *ret;
	__asm__ volatile ("movq	%%gs:%1,%0"
			: "=r" (ret) ,"=m" ((*(volatile u64*) (u64)Offset)));
	return (u64) ret;
}
#endif // _MSC_VER
static inline void* get_kernel32() {
	u8* peb = (u8*)__readgsqword(0x60); // find kernel32 in TEB->PEB->Ldr->ModuleList
	u8* ldr = *(u8**)(peb+0x18);
	u8* module_node = *(u8**)(ldr+0x20);
	u8* dll_base = *(u8**)(module_node+0x20);
	while(dll_base) { // loop through loaded modules looking for kernel32
		u16 dll_name_len = *(u16*)(module_node+0x38);
		u16* dll_name = *(u16**)(module_node+0x40);
		u16* kernel32_name = L"KERNEL32.DLL";
		i32 i = dll_name_len/2 - 1; // length is in bytes, but it's unicode so /2 for num chars
		i32 j = 11; // index of last L in KERNEL32.DLL
		while(j >= 0) { // see if last 11 characters are KERNEL32.DLL (case insensitive)
			if(dll_name[i] != kernel32_name[j] && dll_name[i] != kernel32_name[j] + ('a'-'A'))
				break;
			i--; j--;
		}
		if(j < 0) return dll_base; // found kernel32.dll
		module_node = *(u8**)module_node;
		dll_base = *(u8**)(module_node+0x20);
	}
	return 0; // not found
}

static void* GetProcAddress(u8* handle, char* proc_name, void* (*LoadLibraryA)(char*)) {
	u8* pe_hdr = handle + *(u32*)(handle+0x3C);
	u8* exports_table = handle + *(u32*)(pe_hdr+0x18+0x70);
	u32* names_table = (u32*)(handle + *(u32*)(exports_table+0x20));
	u64 index;
	if(proc_name[0] == '#') { // TODO: test this path
		parse_based_int(proc_name, 1, &index, 10);
	}
	else {
		i64 low = 0;
		i64 high = *(i32*)(exports_table+0x18);
		i64 comparison = 0;
		do {
			if(comparison > 0) low = index + 1;
			if(comparison < 0) high = index - 1;
			index = (high + low) / 2;
			char* cur_proc_name = handle + names_table[index];
			comparison = strcmp(proc_name, cur_proc_name);
		} while(comparison != 0 && low != high);
		if(strcmp(proc_name, handle+names_table[index]) != 0)
			return 0;
	}
	u16* ordinal_table = (u16*)(handle + *(u32*)(exports_table+0x24));
	u32* export_address_table = (u32*)(handle + *(u32*)(exports_table+0x1C));
	u8* addr = handle + export_address_table[ordinal_table[index]];
	u32 exports_table_size = *(u32*)(pe_hdr+0x18+0x70+4);
	if(addr < exports_table || addr >= exports_table + exports_table_size)
		return addr;
	char* forward_name = addr;
	u32 dotIdx = 0;
	while(forward_name[dotIdx] != '.') dotIdx += 1;
	char dll_name[256] = {0};
	memcpy(dll_name, forward_name, dotIdx+1);
	dll_name[dotIdx+1] = 'd';
	dll_name[dotIdx+2] = 'l';
	dll_name[dotIdx+3] = 'l';
	void* other_handle = LoadLibraryA(dll_name);
	return GetProcAddress(other_handle, forward_name+dotIdx+1, LoadLibraryA);
}

void* (*VirtualAlloc)         (void* lpAddress, u64 dwSize, u32 flAllocationType, u32 flProtect);
void* (*GetStdHandle)         (u32 nStdHandle);
void* (*CreateFileA)          (char* lpFileName, u32 dwDesiredAccess, u32 dwShareMode, void* lpSecurityAttributes, u32 dwCreationDisposition, u32 dwFlagsAndAttributes, void* hTemplateFile);
int   (*ReadFile)             (void* hFile, void* lpBuffer, u32 nNumberOfBytesToRead, u32* lpNumberOfBytesRead, void* lpOverlapped);
int   (*WriteFile)            (void* hFile, void* lpBuffer, u32 nNumberOfBytesToWrite, u32* lpNumberOfBytesWritten, void* lpOverlapped);
int   (*CloseHandle)          (void* hObject);
void* (*GetModuleHandleA)     (char* lpModuleName);
void* (*LoadLibraryA)         (char* LoadLibraryA);
u16   (*RegisterClassA)       (void *lpWndClass);
void* (*CreateWindowExA)      (u32,char*,char*,u32,int,int,int,int,void*,void*,void*,void*);
u64   (*DefWindowProcA)       (void* hWnd, u32 Msg, u64 wParam, u64 lParam);
void  (*PostQuitMessage)      (int nExitCode);
i32   (*GetMessageA)          (void* lpMsg,void* hWnd,u32 wMsgFilterMin,u32 wMsgFilterMax);
i32   (*PeekMessageA)         (void* lpMsg,void* hWnd,u32 wMsgFilterMin,u32 wMsgFilterMax,u32 wRemoveMsg);
i32   (*TranslateMessage)     (void *lpMsg);
u64   (*DispatchMessageA)     (void *lpMsg);
int   (*GetFileAttributesExA) (char* lpFileName, int fInfoLevelId, void* lpFileInformation);
char* (*GetCommandLineA)      (void);
void  (*ExitProcess)          (u32 uExitCode);
int   (*GetMonitorInfoA)      (void* hMonitor,void* lpmi);
void* (*MonitorFromPoint)     (i32[2],u32 dwFlags);
void* (*BeginPaint)           (void*, void*);
int   (*EndPaint)             (void*, void*);
int   (*StretchDIBits)        (void* hdc, int xDest, int yDest, int DestWidth, int DestHeight, int xSrc, int ySrc, int SrcWidth, int SrcHeight, void *lpBits, void *lpbmi, u32 iUsage, u32 rop);
void* (*GetDC)                (void*);
int   (*ReleaseDC)            (void*, void*);
void* (*LoadCursorA)          (void* hInstance, char* lpCursorName);
void* (*SetCursor)            (void* hCursor);

static inline void link_win() {
	void* kernel32       = get_kernel32();
	LoadLibraryA         = GetProcAddress(kernel32, "LoadLibraryA", 0);
	VirtualAlloc         = GetProcAddress(kernel32, "VirtualAlloc", LoadLibraryA);
	GetModuleHandleA     = GetProcAddress(kernel32, "GetModuleHandleA", LoadLibraryA);
	GetCommandLineA      = GetProcAddress(kernel32, "GetCommandLineA", LoadLibraryA);
	ExitProcess          = GetProcAddress(kernel32, "ExitProcess", LoadLibraryA);
	GetStdHandle         = GetProcAddress(kernel32, "GetStdHandle", LoadLibraryA);
	GetFileAttributesExA = GetProcAddress(kernel32, "GetFileAttributesExA", LoadLibraryA);
	CreateFileA          = GetProcAddress(kernel32, "CreateFileA", LoadLibraryA);
	ReadFile             = GetProcAddress(kernel32, "ReadFile", LoadLibraryA);
	WriteFile            = GetProcAddress(kernel32, "WriteFile", LoadLibraryA);
	CloseHandle          = GetProcAddress(kernel32, "CloseHandle", LoadLibraryA);
	void* user32         = LoadLibraryA("user32.dll");
	RegisterClassA       = GetProcAddress(user32, "RegisterClassA", LoadLibraryA);
	CreateWindowExA      = GetProcAddress(user32, "CreateWindowExA", LoadLibraryA);
	PostQuitMessage      = GetProcAddress(user32, "PostQuitMessage", LoadLibraryA);
	GetMessageA          = GetProcAddress(user32, "GetMessageA", LoadLibraryA);
	PeekMessageA         = GetProcAddress(user32, "PeekMessageA", LoadLibraryA);
	TranslateMessage     = GetProcAddress(user32, "TranslateMessage", LoadLibraryA);
	DispatchMessageA     = GetProcAddress(user32, "DispatchMessageA", LoadLibraryA);
	DefWindowProcA       = GetProcAddress(user32, "DefWindowProcA", LoadLibraryA);
	GetMonitorInfoA      = GetProcAddress(user32, "GetMonitorInfoA", LoadLibraryA);
	MonitorFromPoint     = GetProcAddress(user32, "MonitorFromPoint", LoadLibraryA);
	BeginPaint           = GetProcAddress(user32, "BeginPaint", LoadLibraryA);
	EndPaint             = GetProcAddress(user32, "EndPaint", LoadLibraryA);
	GetDC                = GetProcAddress(user32, "GetDC", LoadLibraryA);
	ReleaseDC            = GetProcAddress(user32, "ReleaseDC", LoadLibraryA);
	LoadCursorA          = GetProcAddress(user32, "LoadCursorA", LoadLibraryA);
	SetCursor            = GetProcAddress(user32, "SetCursor", LoadLibraryA);
	void* gdi32          = LoadLibraryA("gdi32.dll");
	StretchDIBits        = GetProcAddress(gdi32, "StretchDIBits", LoadLibraryA);
}

//// start common_win.h ////
static inline void* mem_alloc(u64 size) {
	return VirtualAlloc(0, size, 0x2000 | 0x1000, 0x40);
}

static inline u64 file_modified_time(char* filename) {
	u32 fileInfo[9];
	if(!GetFileAttributesExA(filename, 0, &fileInfo)) return 0;
	return ((u64)fileInfo[5] << 32) | fileInfo[6];
}
static inline u64 file_size(char* filename) {
	u32 fileInfo[9];
	if(!GetFileAttributesExA(filename, 0, &fileInfo)) return 0;
	return ((u64)fileInfo[7] << 32) | fileInfo[8];
}
static inline u32 file_read(char* filename, u8* in_buffer, u32 in_size) {
	void* fin = CreateFileA(filename, 0x80000000, 0x00000001, 0, 3, 0, 0);
	if(fin == (void*)-1) return 0;
	u32 bytes_read;
	if(!ReadFile(fin, in_buffer, in_size, &bytes_read, 0)) return 0;
	CloseHandle(fin);
	return bytes_read;
}
static inline u32 file_write(char* filename, u8* out_buffer, u32 out_size) {
	void* fout = CreateFileA(filename, 0x40000000, 0, 0, 2, 0, 0);
	if(fout == (void*)-1) return 0;
	u32 bytes_written;
	if(!WriteFile(fout, out_buffer, out_size, &bytes_written, 0) || bytes_written != out_size) return 0;
	CloseHandle(fout);
	return bytes_written;
}

//// start winnt.h ////
// e_magic and e_lfanew are the only fields used today
typedef struct IMAGE_DOS_HEADER {
	u16 e_magic;    // IMAGE_DOS_SIGNATURE (aka MZ in ASCII)
	u16 e_cblp;     //0x02
	u16 e_cp;       //0x04
	u16 e_crlc;     //0x06
	u16 e_cparhdr;  //0x08
	u16 e_minalloc; //0x0A
	u16 e_maxalloc; //0x0C
	u16 e_ss;       //0x0E
	u16 e_sp;       //0x10
	u16 e_csum;     //0x12
	u16 e_ip;       //0x14
	u16 e_cs;       //0x16
	u16 e_lfarlc;   //0x18
	u16 e_ovno;     //0x1A
	u16 e_res[4];   //0x1C
	u16 e_oemid;    //0x24
	u16 e_oeminfo;  //0x26
	u16 e_res2[10]; //0x28
	u32 e_lfanew;   //0x3C address of IMAGE_NT_HEADERS64 relative to base of image
} IMAGE_DOS_HEADER;

typedef struct IMAGE_FILE_HEADER {
	u16 Machine;
	u16 NumberOfSections;
	u32 TimeDateStamp;        // UNSUED
	u32 PointerToSymbolTable; // UNSUED
	u32 NumberOfSymbols;      // UNSUED
	u16 SizeOfOptionalHeader;
	u16 Characteristics;
} IMAGE_FILE_HEADER;
typedef struct IMAGE_DATA_DIRECTORY {
	u32 VirtualAddress;
	u32 Size;
} IMAGE_DATA_DIRECTORY;
typedef struct IMAGE_OPTIONAL_HEADER64 {
	u16 Magic;
	u8 MajorLinkerVersion;          // UNSUED //0x2
	u8 MinorLinkerVersion;          // UNSUED //0x3
	u32 SizeOfCode;//0x4
	u32 SizeOfInitializedData;      // UNSUED //0x8
	u32 SizeOfUninitializedData;    // UNSUED //0xC
	u32 AddressOfEntryPoint;//0x10
	u32 BaseOfCode;                 // UNSUED//0x14
	u64 ImageBase;//0x18
	u32 SectionAlignment;//0x20
	u32 FileAlignment;//0x24
	u16 MajorOperatingSystemVersion; // UNSUED//0x28
	u16 MinorOperatingSystemVersion; // UNSUED//0x2A
	u16 MajorImageVersion;           // UNSUED//0x2C
	u16 MinorImageVersion;           // UNSUED//0x2E
	u16 MajorSubsystemVersion;//0x30
	u16 MinorSubsystemVersion;       // UNSUED//0x32
	u32 Win32VersionValue;          // UNSUED//0x34
	u32 SizeOfImage;//0x38
	u32 SizeOfHeaders;//0x3C
	u32 CheckSum;                   // UNSUED//0x40
	u16 Subsystem;//0x44
	u16 DllCharacteristics;          // UNSUED//0x46
	u64 SizeOfStackReserve;     // UNSUED//0x48
	u64 SizeOfStackCommit;//0x50
	u64 SizeOfHeapReserve;//0x58
	u64 SizeOfHeapCommit;       // UNSUED//0x60
	u32 LoaderFlags;                // UNSUED//0x68
	u32 NumberOfRvaAndSizes;        // UNSUED//0x6C
	IMAGE_DATA_DIRECTORY DataDirectory[16];//0x70
} IMAGE_OPTIONAL_HEADER64;
typedef struct IMAGE_NT_HEADERS64 {
	u32 Signature;
	IMAGE_FILE_HEADER FileHeader;
	IMAGE_OPTIONAL_HEADER64 OptionalHeader;
} IMAGE_NT_HEADERS64;

// funcAddr = functionsList[ordinalsList[indexOfNameOfFuncInNamesList]]
typedef struct IMAGE_EXPORT_DIRECTORY {
	u32 Characteristics; // UNUSED
	u32 TimeDateStamp;   // UNUSED //0x04
	u16 MajorVersion;     // UNUSED //0x08
	u16 MinorVersion;     // UNUSED //0x0A
	u32 Name; // RVA (relative virtual address) of ASCII name of DLL //0x0C
	u32 Base; // OrdinalBase (unused? see https://stackoverflow.com/a/5654463) //0x10
	u32 NumberOfFunctions; //0x14
	u32 NumberOfNames; //0x18
	u32 AddressOfFunctions; // RVA of list of functions //0x1C
	u32 AddressOfNames; // RVA of list of 32-bit RVAs of ASCII names of functions //0x20
	u32 AddressOfNameOrdinals; // RVA of list of ordinals aka indices into functions list //0x24
} IMAGE_EXPORT_DIRECTORY;
typedef struct IMAGE_SECTION_HEADER {
	u8 Name[8];
	union {
		u32 PhysicalAddress;
		u32 VirtualSize;
	} Misc;
	u32 VirtualAddress;
	u32 SizeOfRawData;
	u32 PointerToRawData;
	u32 PointerToRelocations;
	u32 PointerToLinenumbers;
	u16 NumberOfRelocations;
	u16 NumberOfLinenumbers;
	u32 Characteristics;
} IMAGE_SECTION_HEADER;
//// end winnt.h ////

typedef struct simple_pe simple_pe;
struct simple_pe {
	IMAGE_DOS_HEADER dh;
	IMAGE_NT_HEADERS64 nt;
	IMAGE_SECTION_HEADER textsect;
};

#define CODESIZE (1<<24)
simple_pe default_pe = {
	{ // IMAGE_DOS_HEADER
		0x5A4D,
		0, // UNUSED
		.e_lfanew = sizeof(IMAGE_DOS_HEADER), // e_lfanew (ptr to new header - put right after this one)
	},
	{ // IMAGE_NT_HEADERS64
		0x00004550,
		{ // IMAGE_FILE_HEADER
			0x8664,
			1, // NumberOfSections
			0, // TimeDateStamp
			0, 0, // PointerToSymbolTable, NumberOfSymbols
			sizeof(IMAGE_OPTIONAL_HEADER64), // SizeOfOptionalHeader
			0x0001 | 0x0002 | 0x0020 | 0x0200, // Characteristics
		},
		{ // IMAGE_OPTIONAL_HEADER64
			0x20b,
			0, 0, // MajorLinkerVersion, MinorLinkerVersion
			CODESIZE, // SizeOfCode
			0, 0, // SizeOfInitializedData, SizeOfUninitializedData
			sizeof(simple_pe), // AddressOfEntryPoint
			0x1000, // BaseOfCode (compilers seem to just set this to 0x1000, dunno y)
			0x400000, // ImageBase
			4, 4, // SectionAlignment, FileAlignment
			0, 0, 0, 0, // MajorOperatingSystemVersion, MinorOperatingSystemVersion, MajorImageVersion, MinorImageVersion
			4, 0, // MajorSubsystemVersion, MinorSubsystemVersion
			0, // Win32VersionValue
			sizeof(simple_pe) + CODESIZE, // SizeOfImage
			sizeof(simple_pe), // SizeOfHeaders
			0, // CheckSum
			2, // Subsystem (2=GUI, 3=CUI, 10=EFI)
			0x0400,// | IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE | IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA, // DllCharacteristics
			0x100000,  // SizeOfStackReserve
			0x1000,    // SizeOfStackCommit
			0x1000000, // SizeOfHeapReserve
			0x1000,    // SizeOfHeapCommit
			0, // LoaderFlags
			16, // NumberOfRvaAndSizes
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // DataDirectory array (first is Exports)
		},
	},
	{ // IMAGE_SECTION_HEADER
		'.', 't', 'e', 'x', 't', '\0', '\0', '\0', // Name
		CODESIZE, // VirtualSize
		sizeof(simple_pe), // VirtualAddress
		CODESIZE, // SizeOfRawData
		sizeof(simple_pe), // PointerToRawData
		0, 0, 0, 0, // PointerToRelocations, PointerToLinenumbers, NumberOfRelocations, NumberOfLinenumbers
		0x40000000 | 0x20000000 | 0x00000020, // Characteristics
	}
};
//// end common_win.h ////

//https://courses.cs.washington.edu/courses/cse457/98a/tech/OpenGL/font.c
u8 rasters[][13] = {
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x36, 0x36, 0x36},
{0x00, 0x00, 0x00, 0x66, 0x66, 0xff, 0x66, 0x66, 0xff, 0x66, 0x66, 0x00, 0x00},
{0x00, 0x00, 0x18, 0x7e, 0xff, 0x1b, 0x1f, 0x7e, 0xf8, 0xd8, 0xff, 0x7e, 0x18},
{0x00, 0x00, 0x0e, 0x1b, 0xdb, 0x6e, 0x30, 0x18, 0x0c, 0x76, 0xdb, 0xd8, 0x70},
{0x00, 0x00, 0x7f, 0xc6, 0xcf, 0xd8, 0x70, 0x70, 0xd8, 0xcc, 0xcc, 0x6c, 0x38},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x1c, 0x0c, 0x0e},
{0x00, 0x00, 0x0c, 0x18, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x18, 0x0c},
{0x00, 0x00, 0x30, 0x18, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x18, 0x30},
{0x00, 0x00, 0x00, 0x00, 0x99, 0x5a, 0x3c, 0xff, 0x3c, 0x5a, 0x99, 0x00, 0x00},
{0x00, 0x00, 0x00, 0x18, 0x18, 0x18, 0xff, 0xff, 0x18, 0x18, 0x18, 0x00, 0x00},
{0x00, 0x00, 0x30, 0x18, 0x1c, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x60, 0x60, 0x30, 0x30, 0x18, 0x18, 0x0c, 0x0c, 0x06, 0x06, 0x03, 0x03},
{0x00, 0x00, 0x3c, 0x66, 0xc3, 0xe3, 0xf3, 0xdb, 0xcf, 0xc7, 0xc3, 0x66, 0x3c},
{0x00, 0x00, 0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x78, 0x38, 0x18},
{0x00, 0x00, 0xff, 0xc0, 0xc0, 0x60, 0x30, 0x18, 0x0c, 0x06, 0x03, 0xe7, 0x7e},
{0x00, 0x00, 0x7e, 0xe7, 0x03, 0x03, 0x07, 0x7e, 0x07, 0x03, 0x03, 0xe7, 0x7e},
{0x00, 0x00, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0xff, 0xcc, 0x6c, 0x3c, 0x1c, 0x0c},
{0x00, 0x00, 0x7e, 0xe7, 0x03, 0x03, 0x07, 0xfe, 0xc0, 0xc0, 0xc0, 0xc0, 0xff},
{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xc7, 0xfe, 0xc0, 0xc0, 0xc0, 0xe7, 0x7e},
{0x00, 0x00, 0x30, 0x30, 0x30, 0x30, 0x18, 0x0c, 0x06, 0x03, 0x03, 0x03, 0xff},
{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xe7, 0x7e, 0xe7, 0xc3, 0xc3, 0xe7, 0x7e},
{0x00, 0x00, 0x7e, 0xe7, 0x03, 0x03, 0x03, 0x7f, 0xe7, 0xc3, 0xc3, 0xe7, 0x7e},
{0x00, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x30, 0x18, 0x1c, 0x1c, 0x00, 0x00, 0x1c, 0x1c, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0, 0x60, 0x30, 0x18, 0x0c, 0x06},
{0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x60, 0x30, 0x18, 0x0c, 0x06, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60},
{0x00, 0x00, 0x18, 0x00, 0x00, 0x18, 0x18, 0x0c, 0x06, 0x03, 0xc3, 0xc3, 0x7e},
{0x00, 0x00, 0x3f, 0x60, 0xcf, 0xdb, 0xd3, 0xdd, 0xc3, 0x7e, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xff, 0xc3, 0xc3, 0xc3, 0x66, 0x3c, 0x18},
{0x00, 0x00, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
{0x00, 0x00, 0x7e, 0xe7, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xe7, 0x7e},
{0x00, 0x00, 0xfc, 0xce, 0xc7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc7, 0xce, 0xfc},
{0x00, 0x00, 0xff, 0xc0, 0xc0, 0xc0, 0xc0, 0xfc, 0xc0, 0xc0, 0xc0, 0xc0, 0xff},
{0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xfc, 0xc0, 0xc0, 0xc0, 0xff},
{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xcf, 0xc0, 0xc0, 0xc0, 0xc0, 0xe7, 0x7e},
{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xff, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
{0x00, 0x00, 0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7e},
{0x00, 0x00, 0x7c, 0xee, 0xc6, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06},
{0x00, 0x00, 0xc3, 0xc6, 0xcc, 0xd8, 0xf0, 0xe0, 0xf0, 0xd8, 0xcc, 0xc6, 0xc3},
{0x00, 0x00, 0xff, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0},
{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xdb, 0xff, 0xff, 0xe7, 0xc3},
{0x00, 0x00, 0xc7, 0xc7, 0xcf, 0xcf, 0xdf, 0xdb, 0xfb, 0xf3, 0xf3, 0xe3, 0xe3},
{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xe7, 0x7e},
{0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
{0x00, 0x00, 0x3f, 0x6e, 0xdf, 0xdb, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0x66, 0x3c},
{0x00, 0x00, 0xc3, 0xc6, 0xcc, 0xd8, 0xf0, 0xfe, 0xc7, 0xc3, 0xc3, 0xc7, 0xfe},
{0x00, 0x00, 0x7e, 0xe7, 0x03, 0x03, 0x07, 0x7e, 0xe0, 0xc0, 0xc0, 0xe7, 0x7e},
{0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0xff},
{0x00, 0x00, 0x7e, 0xe7, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
{0x00, 0x00, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
{0x00, 0x00, 0xc3, 0xe7, 0xff, 0xff, 0xdb, 0xdb, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3},
{0x00, 0x00, 0xc3, 0x66, 0x66, 0x3c, 0x3c, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3},
{0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3},
{0x00, 0x00, 0xff, 0xc0, 0xc0, 0x60, 0x30, 0x7e, 0x0c, 0x06, 0x03, 0x03, 0xff},
{0x00, 0x00, 0x3c, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3c},
{0x00, 0x03, 0x03, 0x06, 0x06, 0x0c, 0x0c, 0x18, 0x18, 0x30, 0x30, 0x60, 0x60},
{0x00, 0x00, 0x3c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x3c},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc3, 0x66, 0x3c, 0x18},
{0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x38, 0x30, 0x70},
{0x00, 0x00, 0x7f, 0xc3, 0xc3, 0x7f, 0x03, 0xc3, 0x7e, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xfe, 0xc3, 0xc3, 0xc3, 0xc3, 0xfe, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0},
{0x00, 0x00, 0x7e, 0xc3, 0xc0, 0xc0, 0xc0, 0xc3, 0x7e, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x7f, 0xc3, 0xc3, 0xc3, 0xc3, 0x7f, 0x03, 0x03, 0x03, 0x03, 0x03},
{0x00, 0x00, 0x7f, 0xc0, 0xc0, 0xfe, 0xc3, 0xc3, 0x7e, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x30, 0x30, 0x30, 0x30, 0x30, 0xfc, 0x30, 0x30, 0x30, 0x33, 0x1e},
{0x7e, 0xc3, 0x03, 0x03, 0x7f, 0xc3, 0xc3, 0xc3, 0x7e, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xfe, 0xc0, 0xc0, 0xc0, 0xc0},
{0x00, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x00, 0x18, 0x00},
{0x38, 0x6c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x00, 0x00, 0x0c, 0x00},
{0x00, 0x00, 0xc6, 0xcc, 0xf8, 0xf0, 0xd8, 0xcc, 0xc6, 0xc0, 0xc0, 0xc0, 0xc0},
{0x00, 0x00, 0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x78},
{0x00, 0x00, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xdb, 0xfe, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc6, 0xc6, 0xc6, 0xc6, 0xc6, 0xc6, 0xfc, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x7c, 0xc6, 0xc6, 0xc6, 0xc6, 0xc6, 0x7c, 0x00, 0x00, 0x00, 0x00},
{0xc0, 0xc0, 0xc0, 0xfe, 0xc3, 0xc3, 0xc3, 0xc3, 0xfe, 0x00, 0x00, 0x00, 0x00},
{0x03, 0x03, 0x03, 0x7f, 0xc3, 0xc3, 0xc3, 0xc3, 0x7f, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xe0, 0xfe, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xfe, 0x03, 0x03, 0x7e, 0xc0, 0xc0, 0x7f, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x1c, 0x36, 0x30, 0x30, 0x30, 0x30, 0xfc, 0x30, 0x30, 0x30, 0x00},
{0x00, 0x00, 0x7e, 0xc6, 0xc6, 0xc6, 0xc6, 0xc6, 0xc6, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x18, 0x3c, 0x3c, 0x66, 0x66, 0xc3, 0xc3, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc3, 0xe7, 0xff, 0xdb, 0xc3, 0xc3, 0xc3, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xc3, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0xc3, 0x00, 0x00, 0x00, 0x00},
{0xc0, 0x60, 0x60, 0x30, 0x18, 0x3c, 0x66, 0x66, 0xc3, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0xff, 0x60, 0x30, 0x18, 0x0c, 0x06, 0xff, 0x00, 0x00, 0x00, 0x00},
{0x00, 0x00, 0x0f, 0x18, 0x18, 0x18, 0x38, 0xf0, 0x38, 0x18, 0x18, 0x18, 0x0f},
{0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18},
{0x00, 0x00, 0xf0, 0x18, 0x18, 0x18, 0x1c, 0x0f, 0x1c, 0x18, 0x18, 0x18, 0xf0},
{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x8f, 0xf1, 0x60, 0x00, 0x00, 0x00}
};
#define KEY_MODIFIER_SHIFT (1 << 0)
#define KEY_MODIFIER_CTRL  (1 << 1)
#define KEY_MODIFIER_ALT   (1 << 2)
#define KEY_MODIFIER_FN    (1 << 3)
#define KEY_MODIFIER_SUPER (1 << 4)

// Window Message stuff
#define WM_MOVE 0x0003
#define WM_SIZE 0x0005
#define WM_PAINT 0x000F
#define WM_CLOSE 0x0010
#define WM_DESTROY 0x0002
#define WM_QUIT 0x0012
#define WM_KEYDOWN 0x0100
#define WM_KEYUP 0x0101
#define WM_SYSKEYDOWN 0x0104
#define WM_SYSKEYUP 0x0105

#define WM_MOUSEFIRST 0x0200
#define WM_MOUSEMOVE 0x0200
#define WM_LBUTTONDOWN 0x0201
#define WM_LBUTTONUP 0x0202
#define WM_RBUTTONDOWN 0x0204
#define WM_RBUTTONUP 0x0205
#define WM_MBUTTONDOWN 0x0207
#define WM_MBUTTONUP 0x0208
#define WM_MOUSEWHEEL 0x020A
#define WM_XBUTTONDOWN 0x020B
#define WM_XBUTTONUP 0x020C
#define WM_SETCURSOR 0x020

// mouse xbuttons
#define XBUTTON1 0x0001
#define XBUTTON2 0x0002

// virtual keycodes
#define VK_TAB 0x09
#define VK_RETURN 0x0D
#define VK_SHIFT 0x10
#define VK_CONTROL 0x11
#define VK_MENU 0x12
#define VK_CAPITAL 0x14
#define VK_ESCAPE 0x1B
#define VK_SPACE 0x20
#define VK_END 0x23
#define VK_HOME 0x24
#define VK_LEFT 0x25
#define VK_UP 0x26
#define VK_RIGHT 0x27
#define VK_DOWN 0x28
#define VK_INSERT 0x2D
#define VK_DELETE 0x2E

#define VK_LWIN 0x5B
#define VK_NUMPAD0 0x60
#define VK_NUMPAD1 0x61
#define VK_NUMPAD2 0x62
#define VK_NUMPAD3 0x63
#define VK_NUMPAD4 0x64
#define VK_NUMPAD5 0x65
#define VK_NUMPAD6 0x66
#define VK_NUMPAD7 0x67
#define VK_NUMPAD8 0x68
#define VK_NUMPAD9 0x69
#define VK_MULTIPLY 0x6A
#define VK_ADD 0x6B
#define VK_SEPARATOR 0x6C
#define VK_SUBTRACT 0x6D
#define VK_DECIMAL 0x6E
#define VK_DIVIDE 0x6F
#define VK_F1 0x70
#define VK_F2 0x71
#define VK_F3 0x72
#define VK_F4 0x73
#define VK_F5 0x74
#define VK_F6 0x75
#define VK_F7 0x76
#define VK_F8 0x77
#define VK_F9 0x78
#define VK_F10 0x79
#define VK_F11 0x7A
#define VK_F12 0x7B
#define VK_F13 0x7C
#define VK_F14 0x7D
#define VK_F15 0x7E
#define VK_F16 0x7F
#define VK_F17 0x80
#define VK_F18 0x81
#define VK_F19 0x82
#define VK_F20 0x83
#define VK_F21 0x84
#define VK_F22 0x85
#define VK_F23 0x86
#define VK_F24 0x87
#define VK_NUMLOCK 0x90
#define VK_LSHIFT 0xA0
#define VK_RSHIFT 0xA1
#define VK_LCONTROL 0xA2
#define VK_RCONTROL 0xA3
#define VK_LMENU 0xA4
#define VK_RMENU 0xA5
#define VK_VOLUME_MUTE 0xAD
#define VK_VOLUME_DOWN 0xAE
#define VK_VOLUME_UP 0xAF
#define VK_MEDIA_NEXT_TRACK 0xB0
#define VK_MEDIA_PREV_TRACK 0xB1
#define VK_MEDIA_STOP 0xB2
#define VK_MEDIA_PLAY_PAUSE 0xB3
#define VK_OEM_1 0xBA
#define VK_OEM_PLUS 0xBB
#define VK_OEM_COMMA 0xBC
#define VK_OEM_MINUS 0xBD
#define VK_OEM_PERIOD 0xBE
#define VK_OEM_2 0xBF
#define VK_OEM_3 0xC0
#if _WIN32_WINNT >= 0x0604
#define VK_GAMEPAD_A 0xC3
#define VK_GAMEPAD_B 0xC4
#define VK_GAMEPAD_X 0xC5
#define VK_GAMEPAD_Y 0xC6
#define VK_GAMEPAD_RIGHT_SHOULDER 0xC7
#define VK_GAMEPAD_LEFT_SHOULDER 0xC8
#define VK_GAMEPAD_LEFT_TRIGGER 0xC9
#define VK_GAMEPAD_RIGHT_TRIGGER 0xCA
#define VK_GAMEPAD_DPAD_UP 0xCB
#define VK_GAMEPAD_DPAD_DOWN 0xCC
#define VK_GAMEPAD_DPAD_LEFT 0xCD
#define VK_GAMEPAD_DPAD_RIGHT 0xCE
#define VK_GAMEPAD_MENU 0xCF
#define VK_GAMEPAD_VIEW 0xD0
#define VK_GAMEPAD_LEFT_THUMBSTICK_BUTTON 0xD1
#define VK_GAMEPAD_RIGHT_THUMBSTICK_BUTTON 0xD2
#define VK_GAMEPAD_LEFT_THUMBSTICK_UP 0xD3
#define VK_GAMEPAD_LEFT_THUMBSTICK_DOWN 0xD4
#define VK_GAMEPAD_LEFT_THUMBSTICK_RIGHT 0xD5
#define VK_GAMEPAD_LEFT_THUMBSTICK_LEFT 0xD6
#define VK_GAMEPAD_RIGHT_THUMBSTICK_UP 0xD7
#define VK_GAMEPAD_RIGHT_THUMBSTICK_DOWN 0xD8
#define VK_GAMEPAD_RIGHT_THUMBSTICK_RIGHT 0xD9
#define VK_GAMEPAD_RIGHT_THUMBSTICK_LEFT 0xDA
#endif /* _WIN32_WINNT >= 0x0604 */
#define VK_OEM_4 0xDB
#define VK_OEM_5 0xDC
#define VK_OEM_6 0xDD
#define VK_OEM_7 0xDE
#define VK_OEM_8 0xDF

// custom VKs
#define VK_SEMI_COLON    VK_OEM_1
#define VK_FORWARD_SLASH VK_OEM_2
#define VK_BACK_TICK     VK_OEM_3
#define VK_L_BRACKET     VK_OEM_4
#define VK_BACKSLASH     VK_OEM_5
#define VK_R_BRACKET     VK_OEM_6
#define VK_SINGLE_QUOTE  VK_OEM_7
#define VK_PAGEUP        VK_PRIOR
#define VK_PAGEDOWN      VK_NEXT

i32 view_x = 50;
i32 view_y = 50;
u8 modifiers_held = 0;
u8 mode = 0;
u64 Win32EventHandler(void* window, u32 msg, u64 wp, u64 lp) {
	if(msg == WM_DESTROY || msg == WM_CLOSE)
		ExitProcess(0);
	u64 keycode = wp;
	u64 mousecode = (wp >> 16) & 0xFFFF;
	//u64 was_down = lp & (1 << 30);
	//u64 is_down  = lp & (1 << 31);
	u64 alt_was_down  = lp & (1 << 29);
	if(alt_was_down) modifiers_held |= KEY_MODIFIER_ALT; // better for alt+tab handling
	else             modifiers_held &= ~KEY_MODIFIER_ALT;
	if(msg == WM_SYSKEYDOWN || msg == WM_KEYDOWN) {
		u8 character = 0;
		if(keycode == VK_SHIFT)   modifiers_held |= KEY_MODIFIER_SHIFT;
		if(keycode == VK_CONTROL) modifiers_held |= KEY_MODIFIER_CTRL;
		//if(keycode == VK_MENU)    modifiers_held |= KEY_MODIFIER_ALT; // worse for alt+tab handling

		// text insertion
		if(!modifiers_held) {
			     if(keycode == VK_RETURN)             character = '\n';
			else if(keycode == VK_TAB)                character = '\t';
			else if(keycode == VK_SEMI_COLON)         character = ';';
			else if(keycode == VK_OEM_PLUS)           character = '=';
			else if(keycode == VK_OEM_COMMA)          character = ',';
			else if(keycode == VK_OEM_MINUS)          character = '-';
			else if(keycode == VK_OEM_PERIOD)         character = '.';
			else if(keycode == VK_FORWARD_SLASH)      character = '/';
			else if(keycode == VK_BACK_TICK)          character = '`';
			else if(keycode == VK_L_BRACKET)          character = '[';
			else if(keycode == VK_BACKSLASH)          character = '\\';
			else if(keycode == VK_R_BRACKET)          character = ']';
			else if(keycode == VK_SINGLE_QUOTE)       character = '\'';
			else if(keycode == ' ')                   character = keycode;
			else if(isdigit(keycode))                 character = keycode;
			else if(keycode >= 'A' && keycode <= 'Z') character = keycode + 'a' - 'A';
		}
		if(modifiers_held == KEY_MODIFIER_SHIFT) {
			     if(keycode >= 'A' && keycode <= 'Z') character = keycode;
			else if(keycode == VK_SEMI_COLON)         character = ':';
			else if(keycode == VK_OEM_PLUS)           character = '+';
			else if(keycode == VK_OEM_COMMA)          character = '<';
			else if(keycode == VK_OEM_MINUS)          character = '_';
			else if(keycode == VK_OEM_PERIOD)         character = '>';
			else if(keycode == VK_FORWARD_SLASH)      character = '?';
			else if(keycode == VK_BACK_TICK)          character = '~';
			else if(keycode == VK_L_BRACKET)          character = '{';
			else if(keycode == VK_BACKSLASH)          character = '|';
			else if(keycode == VK_R_BRACKET)          character = '}';
			else if(keycode == VK_SINGLE_QUOTE)       character = '\"';
			else if(keycode == '1')                   character = '!';
			else if(keycode == '2')                   character = '@';
			else if(keycode == '3')                   character = '#';
			else if(keycode == '4')                   character = '$';
			else if(keycode == '5')                   character = '%';
			else if(keycode == '6')                   character = '^';
			else if(keycode == '7')                   character = '&';
			else if(keycode == '8')                   character = '*';
			else if(keycode == '9')                   character = '(';
			else if(keycode == '0')                   character = ')';
		}

		// text deletion
		if(!modifiers_held) {
		}
		if(modifiers_held == KEY_MODIFIER_CTRL) {
			if(keycode == 'H') view_x -= 15*16;
			if(keycode == 'J') view_y += 15*16;
			if(keycode == 'K') view_y -= 15*16;
			if(keycode == 'L') view_x += 15*16;
		}
		if(modifiers_held == (KEY_MODIFIER_CTRL | KEY_MODIFIER_SHIFT)) {
		}
		if(modifiers_held == KEY_MODIFIER_SHIFT) {
			if(keycode == 'H') view_x -= 15*4;
			if(keycode == 'J') view_y += 15*4;
			if(keycode == 'K') view_y -= 15*4;
			if(keycode == 'L') view_x += 15*4;
		}

		// text movement
		if(!(modifiers_held & KEY_MODIFIER_CTRL)) {
			if(keycode == 'H') view_x -= 15;
			if(keycode == 'J') view_y += 15;
			if(keycode == 'K') view_y -= 15;
			if(keycode == 'L') view_x += 15;
		}
		if(modifiers_held & KEY_MODIFIER_CTRL) {
		}
		if(modifiers_held == KEY_MODIFIER_ALT) {
		}
		if((modifiers_held & KEY_MODIFIER_ALT) && (modifiers_held & KEY_MODIFIER_SHIFT)) {
		}

		// editor commands
		if(modifiers_held == KEY_MODIFIER_CTRL) {
			     if(keycode == 'W') { ExitProcess(0); }
			else if(keycode == 'Q') { ExitProcess(0); }
			else if(keycode == 'S') { }
		}
		//arena_gap_splice(&text_arena, new_i, del, &character, !!character);
	}
	if(msg == WM_SYSKEYUP || msg == WM_KEYUP) {
		if(keycode == VK_SHIFT)   modifiers_held &= ~KEY_MODIFIER_SHIFT;
		if(keycode == VK_CONTROL) modifiers_held &= ~KEY_MODIFIER_CTRL;
		//if(keycode == VK_MENU)    modifiers_held &= ~KEY_MODIFIER_ALT;
	}
	if(msg == WM_LBUTTONDOWN) {
		//console_log("l\n", 0);
	}
	if(msg == WM_RBUTTONDOWN) {
		//console_log("r\n", 0);
	}
	if(msg == WM_MBUTTONDOWN) {
		//console_log("m\n", 0);
	}
	if(msg == WM_XBUTTONDOWN) {
		//if(mousecode == XBUTTON1) console_log("b\n", 0);
		//if(mousecode == XBUTTON2) console_log("f\n", 0);
	}
	if(msg == WM_SETCURSOR) {
		//SetCursor(0); // hide cursor (if we wanna do that)
	}
	return DefWindowProcA(window, msg, wp, lp);
}

void* winclass[9] = {0};
void* window;
void* dc;
u32 mi[10] = { 4*10 };
i32 ptZero[2] = { 0, 0 };
u64 msg[6];
i32 bmih[10] = {0};
char str[] = "the quick brown fox jumped over the lazy dog `1234567890-=~!@#$%^&*()_+";
char hex_digits[] = "0123456789ABCDEF";
u8* exe_mem;

int width = 1920;
int height = 1080;
u32* pixel_data;
void render_char(char c, int ix, int iy) {
	for(int y = iy; y < iy + 13; y++) {
		if(y < 0 || y >= height) break;
		for(int x = ix; x < ix + 8; x++) {
			if(x < 0 || x >= width) continue;
			if(rasters[c-0x20][12-(y-iy)] & (1 << (7-(x-ix)))) pixel_data[y * width + x] = -1;
			else pixel_data[y * width + x] = 0;
		}
	}
}
void render_str(char* s, int ix, int iy) {
	while(*s) {
		render_char(*s, ix, iy);
		ix += 10;
		s++;
	}
}
void render_hex_byte(u8 b, int ix, int iy) {
	render_char(hex_digits[(b>>4) & 0xF], ix,    iy);
	render_char(hex_digits[b & 0xF],      ix+10, iy);
}

void _start() {
	link_win();
	// create fullscreen window
	winclass[1] = Win32EventHandler;
	winclass[3] = GetModuleHandleA(0);
	winclass[5] = LoadCursorA(0, (char*)32512); // IDC_ARROW
	winclass[8] = "winclass";
	RegisterClassA(&winclass);
	GetMonitorInfoA(MonitorFromPoint(ptZero, 0x00000001), &mi);
	width = mi[3] - mi[1];
	height = mi[4] - mi[2];
	window = CreateWindowExA(0, winclass[8], "title", 0x80000000|0x10000000,
			mi[1], mi[2], width, height,
			0, 0, winclass[3], 0);
	bmih[0] = 4*10;
	bmih[1] = width;
	bmih[2] = -height;
	bmih[3] = 1 | (32<<16);

	char* file_name = "generated.cubin";
	char* file_name2 = "generated2.cubin";
	int exe_size = file_size(file_name);
	exe_mem = mem_alloc(sizeof(simple_pe) + CODESIZE);
	//memcpy(exe_mem, &default_pe, sizeof(simple_pe));
	file_read(file_name, exe_mem, exe_size);

	// this is for modifying the first IADD3.X generated from ptx_gen.py to see how the assembly changes
	u64* cbs = (u64*)&exe_mem[0x968];
	*cbs &= ~(1 << 19); // Pu=3 bit 83
	*cbs &= ~(1 << 20); // Pv=6 bit 84
	*cbs &= ~(1 << 14); // Pq=5 bit 78
	*cbs &= ~(1 << 16); // !Pq  bit 80
	*cbs |=  (3ULL << 38); // pm_pred bit 38
	file_write(file_name2, exe_mem, exe_size);

	pixel_data = mem_alloc(width*height*4);
	while(1) {
		GetMessageA(&msg, 0, 0, 0); // wait for next input before handling all the rest
		TranslateMessage(&msg);
		DispatchMessageA(&msg);
		while(PeekMessageA(&msg, 0, 0, 0, 0x0001)) {
			TranslateMessage(&msg);
			DispatchMessageA(&msg);
		}
		memset(pixel_data, 0, width*height*4);

		int bpl = 16; // bytes per line
		int line_nums = 1;
		int print_sass = 1;
		for(int i = 0; i < exe_size; i++) {
			int extra_spacing = (i%bpl)/2*5 + (i%bpl)/8*5; // 37 + 9 = 46
			if(line_nums) {
				extra_spacing += 10*10; // 46 + 100 = 146
				if((i % bpl) == 0) {
					for(int j = 0; j < 8; j++) {
						render_char(hex_digits[(i>>((7-j)*4))&0xF], view_x + j*10, view_y + (i/bpl)*15);
					}
				}
			}
			render_hex_byte(exe_mem[i], extra_spacing + view_x + (i%bpl)*20, view_y + (i/bpl)*15);
			// print SASS
			if((i % 16) == 0 && print_sass) {
				u32 ibits = (*(u32*)&exe_mem[i+0] >>  0) & 0xFFF;
				u64 pred  = (*(u64*)&exe_mem[i+0] >> 12) &   0xF;
				u64 dest  = (*(u64*)&exe_mem[i+0] >> 16) &  0xFF;
				u64 src0  = (*(u64*)&exe_mem[i+0] >> 24) &  0xFF;
				u64 src1  = (*(u64*)&exe_mem[i+0] >> 32) &  0xFF;
				u64 nsrc1 = (*(u64*)&exe_mem[i+0] >> 63) &   0x1; // not/neg src1
				u64 src2  = (*(u64*)&exe_mem[i+8] >>  0) &  0xFF;
				u64 nsrc0 = (*(u64*)&exe_mem[i+8] >>  8) &   0x1; // not/neg src0
				u64 cin   = (*(u64*)&exe_mem[i+8] >> 10) &   0x1; // Sc_absolute from nvdisasm strings
				u64 nsrc2 = (*(u64*)&exe_mem[i+8] >> 11) &   0x1; // not/neg src2
				u64 Pq    = (*(u64*)&exe_mem[i+8] >> 13) &   0x7; // 2nd carry in?
				u64 nPq   = (*(u64*)&exe_mem[i+8] >> 16) &   0x1; // !Pq
				u64 coutp = (*(u64*)&exe_mem[i+8] >> 17) &   0x7; // Pu from nvdisasm strings
				u64 Pv    = (*(u64*)&exe_mem[i+8] >> 20) &   0x7; // 2nd carry out?
				u64 cinp  = (*(u64*)&exe_mem[i+8] >> 23) &   0x7; // Pp from nvdisasm strings
				u64 ncinp = (*(u64*)&exe_mem[i+8] >> 26) &   0x1; // !Pp
				if(ibits == 0x210) { // maybe make a lookup table for ibits to mnemonic str index and param switch
					render_str("IADD3", extra_spacing + view_x + 19*20, view_y + (i/bpl)*15);
					if(cin) render_char('X', extra_spacing + view_x + (19+2)*20+10, view_y + (i/bpl)*15);
					if(pred != 7) render_char(hex_digits[pred], extra_spacing + view_x + (19+3)*20, view_y + (i/bpl)*15);
					render_hex_byte(dest, extra_spacing + view_x + (19+4)*20, view_y + (i/bpl)*15);
					char n = '-';
					if(cin) n = '~';
					if(nsrc0) render_char(n, extra_spacing + view_x + (19+6)*20-10, view_y + (i/bpl)*15);
					render_hex_byte(src0, extra_spacing + view_x + (19+6)*20, view_y + (i/bpl)*15);
					if(nsrc1) render_char(n, extra_spacing + view_x + (19+8)*20-10, view_y + (i/bpl)*15);
					render_hex_byte(src1, extra_spacing + view_x + (19+8)*20, view_y + (i/bpl)*15);
					if(nsrc2) render_char(n, extra_spacing + view_x + (19+10)*20-10, view_y + (i/bpl)*15);
					render_hex_byte(src2, extra_spacing + view_x + (19+10)*20, view_y + (i/bpl)*15);
					if(coutp != 7) {
						render_char('P', extra_spacing + view_x + (19+11)*20+10, view_y + (i/bpl)*15);
						render_char(hex_digits[coutp], extra_spacing + view_x + (19+12)*20, view_y + (i/bpl)*15);
					}
					if(cin) {
						if(ncinp) render_char('!', extra_spacing + view_x + (19+13)*20-10, view_y + (i/bpl)*15);
						render_char('P', extra_spacing + view_x + (19+13)*20, view_y + (i/bpl)*15);
						render_char(hex_digits[cinp], extra_spacing + view_x + (19+13)*20+10, view_y + (i/bpl)*15);
					}
					if(Pv != 7) {
						render_char('P', extra_spacing + view_x + (19+15)*20, view_y + (i/bpl)*15);
						render_char(hex_digits[Pv], extra_spacing + view_x + (19+15)*20+10, view_y + (i/bpl)*15);
					}
					if(Pq != 7 || nPq) {
						if(nPq) render_char('!', extra_spacing + view_x + (19+17)*20-10, view_y + (i/bpl)*15);
						render_char('P', extra_spacing + view_x + (19+17)*20, view_y + (i/bpl)*15);
						render_char(hex_digits[Pq], extra_spacing + view_x + (19+17)*20+10, view_y + (i/bpl)*15);
					}
				}
			}
		}

		// draw pixel_data to screen
		dc = GetDC(window);
		StretchDIBits(dc, 0, 0, width, height, 0, 0, width, height, pixel_data, &bmih, 0, 0x00CC0020);
		ReleaseDC(window, dc);
	}
	ExitProcess(0);
}

