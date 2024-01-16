import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial, reduce, wraps
import math
from einops import rearrange

class Conv2dBlock(nn.Module): # padding 卷积+norm+激活 
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='none', pad_type='zero',  use_bias = True,groups=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation == nn.Softmax()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,groups=groups, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
def default(val, default_val):
    return default_val if val is None else val
def cache_method_decorator(cache_attr, cache_namespace, reexecute = False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class newMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2dBlock(in_features,hidden_features,1,1,0,norm='bn',activation='gelu')
        self.dw = Conv2dBlock(hidden_features,hidden_features,3,1,1,norm='bn',activation='gelu',groups=hidden_features)
        self.fc2 = Conv2dBlock(hidden_features,in_features,1,1,0,norm='bn',activation='gelu')
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B,N,C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1,2).reshape(B,C,H,W)
        x = self.fc1(x)
        x = self.dw(x)
        x = self.fc2(x)
        x = x.reshape(B,C,N).transpose(1,2)
        return x
class DwMlp(nn.Module):
    def __init__(self, in_features,Ch, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,window={3:2,5:3,7:3}):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2dBlock(in_features, hidden_features,1,1,0,norm='bn',activation='gelu')
        self.fc2 = Conv2dBlock(hidden_features, out_features,1,1,0,norm='bn',activation='gelu')
        self.drop = nn.Dropout(drop)          
        self.dwconv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = Conv2dBlock(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=cur_window, 
                padding=padding_size,
                stride = 1,
                norm= 'bn', 
                activation= 'gelu',                        
                groups=cur_head_split*Ch,
            )
            self.dwconv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]
    def forward(self, x):
        B,N,C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1,2).reshape(B,C,H,W)             
        x_list = torch.split(x, self.channel_splits, dim=1)                      # Split according to channels.
        conv_x_img_list = [conv(x) for conv, x in zip(self.dwconv_list, x_list)]
        conv_x_img = torch.cat(conv_x_img_list, dim=1)
        x = self.fc2(self.fc1(conv_x_img)).reshape(B,C,H*W).transpose(1,2) 
        return x
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0., n_hashes=4, n_buckets=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.proj = Mlp(self.dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self._cache = {}
    def forward(self, x):
        B, N, C = x.shape
        x_ = self.norm1(x)
        q = self.q(x_).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        rp_x, sum_buckets = self.hash_vectors(self.n_buckets,x_)
        kv = self.kv(rp_x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        k, v = kv[0], kv[1]

        attn = torch.einsum('bHNd,bHnd->bHNn', q, k) * self.scale

        mask_value = -torch.finfo(attn.dtype).max


        mask = sum_buckets.squeeze(-1)>=1
        mask = mask[:,None,None,:].expand_as(attn)
        attn.masked_fill_(~mask, mask_value)
        del mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_ = torch.einsum('bHNn,bHnd->bNHd', attn, v).reshape(B, N,C)
        x = x + self.attn_drop(x_)
        x_ = self.proj(self.norm2(x))
        x = x + self.proj_drop(x_)

        return x
    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            1,
            vecs.shape[-1],
            self.n_hashes,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        rotated_vecs = torch.einsum('btf,bfhi->bhti', vecs, random_rotations)


        if self.n_hashes !=1:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        b,h,n = buckets.shape
    
        buckets = torch.zeros(b,h,n,n_buckets,device=device).scatter_(3,buckets.unsqueeze(-1),1)
        sum_buckets = torch.sum(buckets,dim=2).unsqueeze(-1)
        random_pool_x = torch.einsum('bnf,bhni->bhif', vecs, buckets)
        random_pool_x = random_pool_x/(sum_buckets.detach() + 1e-20)
        random_pool_x = random_pool_x.view(b,h*n_buckets,-1)
        sum_buckets = sum_buckets.view(b,h*n_buckets,1)
        return random_pool_x, sum_buckets.detach()
    
class newBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0.,  n_buckets=8,crpe=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        
        buckets_dic={n_buckets*2:2,n_buckets:3,n_buckets//2:3}
        self.z = nn.Linear(dim, n_buckets*2*2 + n_buckets*3 + n_buckets//2*3 , bias=qkv_bias)
        self.head_splits = []
        self.channel_splits = []
        for buckets, num in buckets_dic.items():                                                             # Use dilation=1 at default.
            self.head_splits.append(num)
            self.channel_splits.append(num*buckets) 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.proj = DwMlp(self.dim,Ch=head_dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self.crpe = crpe
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_ = self.norm1(x)
        qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]         
        z_list = torch.split(self.z(x_), self.channel_splits, dim=2)                      
        z_list = [z.reshape(B,N,heads_num,-1).permute(0,2,1,3).softmax(dim=2)
                            for z, heads_num in zip(z_list,self.head_splits)]
        q_list = torch.split(q, self.head_splits, dim=1)    
        k_list = torch.split(k, self.head_splits, dim=1)    
        v_list = torch.split(v, self.head_splits, dim=1)    
        k_list = [torch.einsum('bhNd,bhNn->bhnd', k, z) for k,z in zip(k_list,z_list)]
        v_list = [torch.einsum('bhNd,bhNn->bhnd', v, z) for v,z in zip(v_list,z_list)]
        attn_list = [torch.einsum('bhNd,bhnd->bhNn', q, k* self.scale).softmax(dim=-1)  for q,k in zip(q_list,k_list)]
        x_list = [torch.einsum('bhNn,bhnd->bhNd', attn, v).transpose(1,2).reshape(B, N,-1) for attn, v in zip(attn_list,v_list)]
        x = x + self.attn_drop(torch.cat(x_list,dim=2)) + self.crpe(q,v,size=[H,W])
        x_ = self.proj(self.norm2(x))
        x = x + self.proj_drop(x_)

        return x

class newdeBlock(nn.Module):
    def __init__(self, dim,dedim, num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0.,  n_buckets=8,crpe=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        buckets_dic={n_buckets*2:2,n_buckets:3,n_buckets//2:3}
        self.z = nn.Linear(dedim, n_buckets*2*2 + n_buckets*3 + n_buckets//2*3 , bias=qkv_bias)
        self.head_splits = []
        self.channel_splits = []
        for buckets, num in buckets_dic.items():                                                             # Use dilation=1 at default.
            self.head_splits.append(num)
            self.channel_splits.append(num*buckets) 
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dedim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm1_1 = nn.LayerNorm(dedim)
        self.proj = DwMlp(self.dim,Ch=head_dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self.crpe = crpe
    def forward(self, x):
        x,dx=x[0],x[1]
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_ = self.norm1(x)
        dx_ = self.norm1_1(dx)
        kv = self.kv(dx_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        k, v = kv[0], kv[1] 
        q = self.q(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute( 0,  2, 1, 3)
        z_list = torch.split(self.z(dx_), self.channel_splits, dim=2)                      
        z_list = [z.reshape(B,N//4,heads_num,-1).permute(0,2,1,3).softmax(dim=2)
                            for z, heads_num in zip(z_list,self.head_splits)]
        q_list = torch.split(q, self.head_splits, dim=1)    
        k_list = torch.split(k, self.head_splits, dim=1)    
        v_list = torch.split(v, self.head_splits, dim=1)    
        k_list = [torch.einsum('bhNd,bhNn->bhnd', k, z) for k,z in zip(k_list,z_list)]
        v_list = [torch.einsum('bhNd,bhNn->bhnd', v, z) for v,z in zip(v_list,z_list)]
        attn_list = [torch.einsum('bhNd,bhnd->bhNn', q, k* self.scale).softmax(dim=-1)  for q,k in zip(q_list,k_list)]
        x_list = [torch.einsum('bhNn,bhnd->bhNd', attn, v).transpose(1,2).reshape(B, N,-1) for attn, v in zip(attn_list,v_list)]
        x = x + self.attn_drop(torch.cat(x_list,dim=2)) + self.crpe(q,v,size=[H,W])
        x_ = self.proj(self.norm2(x))
        x = x + self.proj_drop(x_)

        return [x,dx]

class newnewBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                  attn_drop=0., proj_drop=0., n_hashes=4, n_buckets=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.z = nn.Linear(dim, n_buckets*self.num_heads, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.norm1 = nn.LayerNorm(self.dim)
        self.proj = newMlp(self.dim,hidden_features=4*self.dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(self.dim)
        # self.fuse1 =  Conv2dBlock(2*self.dim,2*self.dim,3,1,1,norm='bn',activation='gelu',groups=2*self.dim)
        # self.fuse2 = nn.Linear(2*self.dim,self.dim)
        self._cache = {}
        self.window_size = 16
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        self.window_size = H//2
        x_ = self.norm1(x)
        qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        z = self.z(x_).reshape(B, -1, self.num_heads, self.n_buckets).permute(0, 2, 1, 3)
        z_= z.softmax(dim=2)

        k_ = torch.einsum('bhNd,bhNn->bhnd', k, z_) 
        v_ = torch.einsum('bhNd,bhNn->bhnd', v, z_)
        attn_ = torch.einsum('bhNd,bhnd->bhNn', q, k_) * self.scale
        attn_ = self.attn_drop(attn_.softmax(dim=-1))
        x_ = torch.einsum('bhNn,bhnd->bhNd', attn_, v_).transpose(1,2).reshape(B, N,C)

        # k__ = F.avg_pool2d(k.transpose(2,3).reshape(B,C,H,W),kernel_size=8,stride=8)\
        #     .reshape(B,self.num_heads,C// self.num_heads,-1).transpose(2,3) 
        # v__ = F.avg_pool2d(v.transpose(2,3).reshape(B,C,H,W),kernel_size=8,stride=8)\
        #     .reshape(B,self.num_heads,C// self.num_heads,-1).transpose(2,3) 
        # attn__ = torch.einsum('bhNd,bhnd->bhNn', q, k__) * self.scale
        # attn__ = self.attn_drop(attn__.softmax(dim=-1))
        # x__ = torch.einsum('bhNn,bhnd->bhNd', attn__, v__).transpose(1,2).reshape(B, N,C)
        ### window 
        
        # z__ = self.window_partition(z,B,H,W,self.window_size).softmax(dim=3)
        # k__ = self.window_partition(k,B,H,W,self.window_size)
        # v__ = self.window_partition(v,B,H,W,self.window_size)
        # q__ = self.window_partition(q,B,H,W,self.window_size)
        # k__ =  torch.einsum('bhIwd,bhIwn->bhInd', k__, z__)   
        # v__ =  torch.einsum('bhIwd,bhIwn->bhInd', v__, z__)       
        # attn__ = torch.einsum('bhIwd,bhInd->bhIwn', q__, k__) * self.scale
        # attn__ = self.attn_drop(attn__.softmax(dim=-1))
        # x__ = torch.einsum('bhIwn,bhInd->bhIwd', attn__, v__)
        # x__ = x__.view(B,self.num_heads,H//self.window_size,
        #     W//self.window_size,self.window_size,self.window_size,-1).permute(0,2,4,3,5,1,6).contiguous().view(B,N,C)
        
        ### shifted window
        # x_ = self.fuse1(torch.cat((x_,x__),dim=-1).transpose(1,2).reshape(B,-1,H,W))
        # x = x + self.attn_drop(self.fuse2(x_.reshape(B,-1,H*W).transpose(1,2)))
        # x1 = self.local_cluster(q,z,k,v, B,H,W,kernel_size=7)
        # v_pool = F.avg_pool2d(v.transpose(2,3).reshape(B,-1,H,W),kernel_size=7,padding=7//2,stride=1,count_include_pad=False)
        # v_pool = v_pool.view(B,-1,H*W).transpose(1,2)
        x = x + self.attn_drop(x_) 
        x_ = self.proj(self.norm2(x))

        x = x + self.proj_drop(x_)  

        return x
    # def forward(self, x):
    #     B, N, C = x.shape
    #     H = W = int(math.sqrt(N))
    #     self.window_size = H//2
    #     x_ = self.norm1(x)
    #     qkv = self.qkv(x_).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0,  3, 1, 4)
    #     # q, k, v = qkv[0], qkv[1], qkv[2]
    #     z = self.z(x_).reshape(B, -1, 2,self.num_heads, self.n_buckets).permute(2, 0,  3, 1, 4)
    #     z, zz= z[0].softmax(dim=2),z[1].softmax(dim=-1)

    #     qkv_ = torch.einsum('lbhNd,bhNn->lbhnd', qkv, z) 
    #     q_, k_, v_ = qkv_[0], qkv_[1], qkv_[2] 
    #     attn_ = torch.einsum('bhnd,bhNd->bhnN', q_, k_) * self.scale
    #     attn_ = self.attn_drop(attn_.softmax(dim=-1))
    #     x_ = torch.einsum('bhNn,bhnd->bhNd', attn_, v_)

    #     x_ = torch.einsum('bhNn,bhnd->bhNd', zz, x_).transpose(1,2).reshape(B, -1,C)
    
    #     x = x + self.attn_drop(x_) 
    #     x_ = self.proj(self.norm2(x))

    #     x = x + self.proj_drop(x_)  

    #     return x
    def local_cluster(self,q,z,k,v, B,H,W,kernel_size):
        n = z.shape[-1]
        d = k.shape[-1]
        z = z.transpose(2,3).reshape(B,-1,H,W)
        z_pool = F.avg_pool2d(z, kernel_size=kernel_size,
                              padding=kernel_size//2,stride=1,count_include_pad=False)
        z = (z/(z_pool+1e-16)).reshape(B,self.num_heads,n,1,H,W)
        z = z.reshape(B,self.num_heads,n,1,H,W)
        k = k.transpose(2,3).reshape(B,self.num_heads,1,d,H,W)
        kz = torch.einsum('bhnlHW,bhldHW->bhndHW', z, k).reshape(B,-1,H,W) 
        kz = F.avg_pool2d(kz, kernel_size=kernel_size,
                              padding=kernel_size//2,stride=1,count_include_pad=False)
        
        v = v.transpose(2,3).reshape(B,self.num_heads,1,d,H,W)
        vz = (v*z).reshape(B,-1,H,W)
        v_pool = F.avg_pool2d(vz, kernel_size=kernel_size,
                              padding=kernel_size//2,stride=1,count_include_pad=False) 
        
        k_pool = k_pool.reshape(B,self.num_heads,n,d,-1).permute(0,1,4,2,3)
        v_pool = v_pool.reshape(B,self.num_heads,n,d,-1).permute(0,1,4,2,3)
        q = q[:,:,:,None,:]
        attn = torch.einsum('bhNld,bhNnd->bhNln', q, k_pool) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bhNln,bhNnd->bhNld', attn, v_pool).unsqueeze(3).transpose(1,2).reshape(B,H*W,-1)
        return x
    def window_partition(self,x, B,H,W,window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        z__ = x.reshape(B,self.num_heads,H//window_size,window_size,W//window_size,window_size,x.shape[-1])
        z__ = z__.permute(0,1,2,4,3,5,6).contiguous().view(B,self.num_heads,
                -1,window_size**2,x.shape[-1])
        return z__




class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, window={3:2,5:3,7:3}):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                       It will apply different window size to the attention head splits.
        """
        super().__init__()
         
        
        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1                                                                 # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2         # Determine padding size. Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
            cur_conv = nn.Conv2d(cur_head_split*Ch, cur_head_split*Ch,
                kernel_size=(cur_window, cur_window), 
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),                          
                groups=cur_head_split*Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x*Ch for x in self.head_splits]

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size
        B, h, Nv, Ch = v.shape
        assert N == H * W

        # Convolutional relative position encoding.
        q_img = q                                                            # Shape: [B, h, H*W, Ch].
        v_img = v  
        Hv =   int(math.sqrt(Nv))                                                         # Shape: [B, h, H*W, Ch].
        
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=Hv, W=Hv)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        if Hv<H:
            v_img = F.interpolate(v_img,size=(H,W),mode='bilinear',align_corners=False)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)                      # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img


        return EV_hat_img.transpose(1,2).flatten(2)
class Clusterformer(nn.Module):

    def __init__(self, in_chans=3, n_class=1, 
                 depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], drop_path_rate=0.0, 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        dim = 512
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)


        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.posembed_layers = nn.ModuleList() 
        self.dstages = nn.ModuleList()
        for i in range(4):
            rposembed_layer = ConvRelPosEnc(dims[i]//8)
            stage = nn.Sequential(
                *[newBlock(dim=dims[i], num_heads=8,n_buckets=2**(7-i)*64//dims[i],crpe=rposembed_layer) for j in range(depths[i])]
            )
            self.stages.append(stage)
            if i<3:
                dstage = nn.Sequential(
                    *[newdeBlock(dim=dims[i],dedim=dims[i+1], num_heads=8,n_buckets=2**(7-i)*64//dims[i+1],crpe=rposembed_layer) for j in range(2)]
                )
                self.dstages.append(dstage)
            cur += depths[i]
            posembed_layer = nn.Conv2d(dims[i],dims[i],3,1,1,groups=dims[i])
            self.posembed_layers.append(posembed_layer)
        self.dims = dims

        self.class3 = nn.Conv2d(dims[3], n_class, 1, 1, 0)
        self.class2 = nn.Conv2d(dims[2], n_class, 1, 1, 0)
        self.class1 = nn.Conv2d(dims[1], n_class, 1, 1, 0)
        self.class0 = nn.Conv2d(dims[0], n_class,1,1,0)
        self.apply(self._init_weights)
        self.dims = dims
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None: 
                nn.init.constant_(m.bias, 0)
    def forward_features(self, x):
        y = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            B,C,H,W = x.shape
            x = x + self.posembed_layers[i](x)
            x = x.reshape(B,C,H*W).transpose(1,2) 
            x = self.stages[i](x)
            x = x.transpose(1,2).reshape(B,C,H,W)
            y.append(x)
        return y # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        y = self.forward_features(x)
        # y3 = self.class3(y[3])
        for i in range(2,-1,-1):
            B,C,H,W = y[i].shape
            xx = y[i].reshape(B,-1,H*W).transpose(1,2) 
            dx = y[i+1].reshape(B,-1,H*W//4).transpose(1,2) 
            y[i],_ = self.dstages[i]([xx,dx])
            y[i] = y[i].transpose(1,2).reshape(B,C,H,W)
  
        # fpn0 = self.fpn0(l0)
        y0 = self.class0(y[0])
        # y1 = self.class1(y[1])
        # y2 = self.class2(y[2])


        # Classifier
        # y = self.class_final(torch.cat((fpn0,fpn1,fpn2,fpn3),dim=1))
        y = F.interpolate(y0,size=x.size()[2:], mode='bilinear', align_corners=False)       

        return y #,y0,y1,y2,y3
class old_Clusterformer(nn.Module):

    def __init__(self, in_chans=3, n_class=1, 
                 depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], drop_path_rate=0.0, 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        dim = 512
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)


        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.posembed_layers = nn.ModuleList() 
        for i in range(4):
            rposembed_layer = ConvRelPosEnc(dims[i]//8)
            stage = nn.Sequential(
                *[newBlock(dim=dims[i], num_heads=8,n_buckets=2**(7-i)*64//dims[i],crpe=rposembed_layer) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            posembed_layer = nn.Conv2d(dims[i],dims[i],3,1,1,groups=dims[i])
            self.posembed_layers.append(posembed_layer)
        self.dims = dims
        self.lateral0 = Conv2dBlock(dims[0], dim, 3, 1, 1,norm='bn',activation='gelu')
        self.lateral1 = Conv2dBlock(dims[1], dim, 3, 1, 1,norm='bn',activation='gelu')
        self.lateral2 = Conv2dBlock(dims[2], dim, 3, 1, 1,norm='bn',activation='gelu')
        self.lateral3 = Conv2dBlock(dims[3], dim, 3, 1, 1,norm='bn',activation='gelu')
        self.dappm = DAPPM(512,512,512)
        self.class3 = nn.Conv2d(dim, n_class, 3, 1, 1)
        self.class2 = nn.Conv2d(dim, n_class, 3, 1, 1)
        self.class1 = nn.Conv2d(dim, n_class, 3, 1, 1)
        self.class0 = nn.Conv2d(dim, n_class,3,1,1)
        self.apply(self._init_weights)
        self.dims = dims
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None: 
                nn.init.constant_(m.bias, 0)
    def forward_features(self, x):
        y = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            B,C,H,W = x.shape
            x = x + self.posembed_layers[i](x)
            x = x.reshape(B,C,H*W).transpose(1,2) 
            x = self.stages[i](x)
            x = x.transpose(1,2).reshape(B,C,H,W)
            y.append(x)
        return y # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        y = self.forward_features(x)
        l3 = self.lateral3(y[3])
        l3 = self.dappm(l3)
        l2 = self.lateral2(y[2]) + F.interpolate(l3,size=y[2].size()[2:],  mode='bilinear', align_corners=False)
        l1= self.lateral1(y[1]) + F.interpolate(l2,size=y[1].size()[2:],  mode='bilinear', align_corners=False)
        l0= self.lateral0(y[0]) + F.interpolate(l1,size=y[0].size()[2:],  mode='bilinear', align_corners=False)
        # fpn0 = self.fpn0(l0)
        y0 = self.class0(l0)
        y1 = self.class1(l1)
        y2 = self.class2(l2)
        y3 = self.class3(l3)

        # Classifier
        # y = self.class_final(torch.cat((fpn0,fpn1,fpn2,fpn3),dim=1))
        y = F.interpolate(y0,size=x.size()[2:], mode='bilinear', align_corners=False)       

        return y #,y0,y1,y2,y3    
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim = 1):
        t = torch.arange(x.shape[seq_dim], device = x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :].type_as(x)
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AdaptiveAvgPool2d(2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AdaptiveAvgPool2d(3),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AdaptiveAvgPool2d(6),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=False)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 

