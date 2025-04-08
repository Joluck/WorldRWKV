########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from typing import Optional
import types, gc, os, time, re, math
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
current_path = os.path.dirname(os.path.abspath(__file__))
ROCm_flag=torch.version.hip is not None
########################################################################################################

if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    def __nop(ob):
        return ob
    MyFunction = __nop
    MyStatic = __nop

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    if ROCm_flag == True:
        try:
            load(
                name=f"wkv_cuda",
                sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
                verbose=True,
                extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
                extra_cuda_cflags=['-xhip', '-fopenmp', '-ffast-math', '-O3','-munsafe-fp-atomics','--offload-arch=gfx1100'],
                is_python_module=False)
            DISABLE_CUBLAS_GEMM = False
        except:
            print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
            load(
                name=f"wkv_cuda",
                sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
                verbose=True,
                extra_cuda_cflags=['-xhip', '-fopenmp', '-ffast-math', '-O3','-munsafe-fp-atomics','--offload-arch=gfx1100'],
                extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
                is_python_module=False)
            DISABLE_CUBLAS_GEMM = True
    else:
        try:
            load(
                name=f"wkv_cuda",
                sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
                verbose=True,
                extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
                extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
                is_python_module=False)
            DISABLE_CUBLAS_GEMM = False
        except:
            print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
            load(
                name=f"wkv_cuda",
                sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
                verbose=True,
                extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
                extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
                is_python_module=False)
            DISABLE_CUBLAS_GEMM = True      
    @MyStatic
    def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
        assert 1 * C % min(C, 32) == 0
        assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
        assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        return y, aa, bb, pp
    @MyStatic
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        return y
    @MyStatic
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (N,)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        return y.to(dtype=x.dtype)
else:
    os.environ["RWKV_CUDA_ON"] = '0'


@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

if os.environ.get('RWKV_CUDA_ON') == '1':
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda' and x.dtype == torch.float16:
            B, N, M = x.shape[0], w.shape[0], w.shape[1]
            return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_seq(x, w, mx, rx, my, ry)
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda':
            N, M = w.shape[0], w.shape[1]
            return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_one(x, w, mx, rx, my, ry)
else:
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        return torch_mm8_seq(x, w, mx, rx, my, ry)
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        return torch_mm8_one(x, w, mx, rx, my, ry)

def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
    if len(x.shape) == 1:
        return mm8_one(x, w, mx, rx, my, ry)
    return mm8_seq(x, w, mx, rx, my, ry)

def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")


if os.environ.get('RWKV_CUDA_ON') == '1' and not DISABLE_CUBLAS_GEMM:
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        if output_dtype is None:
            output_dtype = a.dtype
        if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
            if len(a.shape) == 1:
                assert len(b.shape) == 2
                c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                a = a.unsqueeze(0)
            else:
                assert len(a.shape) == len(b.shape)
                assert len(a.shape) == 2 or len(a.shape) == 3
                # torch.empty((*a.shape[:-1], b.shape[-1])) doesn't work with jit
                if len(a.shape) == 2:
                    c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
                else:
                    c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
            torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
            return c
        else:
            return (a @ b).to(output_dtype)

else:
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        return (a @ b).to(output_dtype)


if os.environ.get('RWKV_DML_ON') == '1':
    import torch_directml
    print("PyTorch with DirectML Enabled")


print(f'\n### RWKV-7 "Goose" enabled ###\n')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script
from typing import List

DTYPE = None
DEVICE = None
HEAD_SIZE = 64

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    if ROCm_flag == True:
        load(name="wkv7s", sources=[f"{current_path}/cuda/rwkv7_op.cpp", f"{current_path}/cuda/rwkv7.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=['-xhip', '-fopenmp', '-ffast-math', '-O3','-munsafe-fp-atomics','--offload-arch=gfx1100',f"-D_N_={HEAD_SIZE}"])
    else:
        load(name="wkv7s", sources=[f"{current_path}/cuda/rwkv7_op.cpp", f"{current_path}/cuda/rwkv7.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
                y = torch.empty((T, C), device=DEVICE, dtype=r.dtype, requires_grad=False, memory_format=torch.contiguous_format)

                if DTYPE == torch.float16:
                    torch.ops.wkv7s.forward_fp16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif DTYPE == torch.bfloat16:
                    torch.ops.wkv7s.forward_bf16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif DTYPE == torch.float32:
                    torch.ops.wkv7s.forward_fp32(1, T, C, H, state, r, w, k, v, a, b, y)

                return y
    def RWKV7_OP(state, r, w, k, v, a, b):
        return WKV_7.apply(state, r, w, k, v, a, b)

########################################################################################################

class RWKV(MyModule):
    def __init__(self, model, strategy):
        global DTYPE, DEVICE
        super().__init__()
        self.eval()
        args = types.SimpleNamespace()
        self.args = args
        # args.MODEL_NAME = model

        # print(f'Loading {model} ({strategy})\n')

        ss = strategy.split(' ')
        DEVICE = ss[0]
        if ss[1] == 'fp16':
            DTYPE = torch.half
        elif ss[1] == 'fp32':
            DTYPE = torch.float32
        elif ss[1] == 'bf16':
            DTYPE = torch.bfloat16
        else:
            assert False, "currently rwkv7 strategy must be: cuda/cpu fp16/fp32/bf16"
        
        # self.z = torch.load(args.MODEL_NAME + '.pth', map_location=DEVICE)
        self.z = model

        # for k,v in self.z.items():
        #     print(k, v.shape)
        z = self.z

        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.head_size = self.head_size
        args.vocab_size, args.n_embd = z['emb.weight'].shape

        args.n_layer = 0
        keys = list(z.keys())
        for k in keys:
            layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            args.n_layer = max(args.n_layer, layer_id+1)
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()

        self.n_embd = args.n_embd
        self.n_layer = args.n_layer

        # z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def forward(self, idx, state, full_output=False, sign=None):
        if state == None:
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
                state[i*3+1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
                state[i*3+2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)

        x = self.z['emb.weight'][idx]
        if isinstance(sign, torch.Tensor):
            sign = sign.squeeze(0)
            # sign = F.layer_norm(sign, (self.args.n_embd,), weight=self.z['blocks.0.ln0.weight'], bias=self.z['blocks.0.ln0.bias'])

            x = torch.cat((sign,x.to('cuda')), dim=0)

        x = F.layer_norm(x, (self.args.n_embd,), weight=self.z['blocks.0.ln0.weight'], bias=self.z['blocks.0.ln0.bias'])
        # if isinstance(sign, torch.Tensor):
        #     print(x)

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(x, state, full_output)
            else:
                return self.forward_one(x[0], state)
        else:
            return self.forward_one(x, state)

    @MyFunction
    def forward_one(self, x, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            #x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
                # if math.isnan(torch.min(x).item()): print(idx, i)

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @MyFunction
    def forward_seq(self, x, state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            #x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

if os.environ.get('RWKV_CUDA_ON') == '1':
    @MyStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
        xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first
else:
    @MyStatic
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
        T = x.shape[0]
        xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
        xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

        r = xr @ R_
        w = torch.tanh(xw @ w1) @ w2
        k = xk @ K_
        v = xv @ V_
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
            vk = v_.view(H,N,1) @ k_.view(H,1,N)
            ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
            state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
            xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]

########################################################################################################



