#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<iostream>
#include<cuda_runtime.h>
#include<cstring>
#include<vector>
#include<memory>
#include<stdexcept>
#include<cassert>
#include<random>
#include<iomanip>
#include<cmath>
#include<curand_kernel.h>
#include<cublas_v2.h>
#include<cfloat>
#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/transform.h>
#include<tuple>

using namespace std;
namespace py = pybind11;

//define
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N)
{
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNAL_LOOP(i, n)\
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;i<n;i += blockDim.x * gridDim.x)


//tensor
template<typename T>
class Tensor_template
{
    public:

    struct CudaDeleter
    {
        void operator()(T* ptr) const
        {
            cudaFree(ptr);
        }
    };
    string name;
    unique_ptr<T[]> cpu_ptr;
    unique_ptr<T[], CudaDeleter> gpu_ptr;
    unique_ptr<T[]> cpu_grad_ptr;
    unique_ptr<T[], CudaDeleter> gpu_grad_ptr;
    vector<int> shape;
    int num_dim = 1;
    int len = 1;
    string device;
    bool requires_grad = 1;

    Tensor_template(const string& name, const vector<int>& shape, const string& device, bool requires_grad = 1)
    {
        this->name = name;
        this->shape = shape;
        this->device = device;
        this->num_dim = shape.size();
        this->requires_grad = requires_grad;
        for(int x: shape)
        {
            assert(x > 0);
            len *= x;
        }

        if(device == "cpu")
        {
            cpu_ptr.reset(new T[len]());
            if(requires_grad)
            {
                cpu_grad_ptr.reset(new T[len]());
            }   
        }
        else if(device == "gpu")
        {
            T* p;
            cudaError_t err = cudaMalloc((void**)&p, len * sizeof(T));
            assert(err == cudaSuccess);
            cudaMemset(p, 0, len * sizeof(T));
            gpu_ptr.reset(p);
            if(requires_grad)
            {
                T* q;
                cudaError_t err = cudaMalloc((void**)&q, len * sizeof(T));
                assert(err == cudaSuccess);
                cudaMemset(q, 0, len * sizeof(T));
                gpu_grad_ptr.reset(q);
            }
        }
        else
        {
            assert(0);
        }
    }

    void reshape(const vector<int>& input)
    {
        int total = 1;
        for(int k: input)
        {
            total *= k;
        }
        assert(total == len);
        shape = input;
    }

    void from_numpy(const py::array_t<T>& np)
    {
        py::buffer_info buf = np.request();

        if(np.ndim() < 1) throw invalid_argument("Numpy array must have dimension > 1 !!!!!!!!!!!!!");
        if(shape.size() != np.ndim()) throw invalid_argument("Numpy array must have same dimension numbers with Tensor !!!!!!!!!!");
        for(int i=0; i<shape.size(); i++)
        {
            if(shape[i] != np.shape(i)) throw invalid_argument("Numpy array must have same shape with Tensor !!!!!!!!!!!!!");
        }

        bool from_gpu = 0;
        if(device == "gpu") 
        {
            cpu();
            from_gpu = 1;
        }
        memcpy(cpu_ptr.get(), static_cast<T*>(buf.ptr), len * sizeof(T));
        if(from_gpu) gpu();
    }

    void grad_from_numpy(const py::array_t<T>& np)
    {
        py::buffer_info buf = np.request();

        if(np.ndim() < 1) throw invalid_argument("Numpy array must have dimension > 1 !!!!!!!!!!!!!");
        if(shape.size() != np.ndim()) throw invalid_argument("Numpy array must have same dimension numbers with Tensor !!!!!!!!!!");
        for(int i=0; i<shape.size(); i++)
        {
            if(shape[i] != np.shape(i)) throw invalid_argument("Numpy array must have same shape with Tensor !!!!!!!!!!!!!");
        }

        bool from_gpu = 0;
        if(device == "gpu") 
        {
            cpu();
            from_gpu = 1;
        }
        memcpy(cpu_grad_ptr.get(), static_cast<T*>(buf.ptr), len * sizeof(T));
        if(from_gpu) gpu();
    }

    py::array_t<T> to_numpy()
    {
        py::array_t<T> res(len);
        py::buffer_info buf = res.request();
        bool from_gpu = 0;
        if(device == "gpu")
        {
            from_gpu = 1;
            cpu();
        }
        memcpy(static_cast<T*>(buf.ptr), cpu_ptr.get(), len * sizeof(T));
        if(from_gpu) gpu();
        return res.reshape(shape);
    }

    py::array_t<T> grad_to_numpy()
    {
        py::array_t<T> res(len);
        py::buffer_info buf = res.request();
        bool from_gpu = 0;
        if(device == "gpu")
        {
            from_gpu = 1;
            cpu();
        }
        memcpy(static_cast<T*>(buf.ptr), cpu_grad_ptr.get(), len * sizeof(T));
        if(from_gpu) gpu();
        return res.reshape(shape);
    }

    vector<int> size()
    {
        return shape;
    }

    int length()
    {
        return len;
    }

    void cpu()
    {
        assert(device == "gpu");
        cpu_ptr.reset(new T[len]());
        cudaMemcpy(cpu_ptr.get(), gpu_ptr.get(), len * sizeof(T), cudaMemcpyDeviceToHost);
        gpu_ptr.reset(nullptr);
        if(requires_grad) 
        {   
            cpu_grad_ptr.reset(new T[len]());
            cudaMemcpy(cpu_grad_ptr.get(), gpu_grad_ptr.get(), len * sizeof(T), cudaMemcpyDeviceToHost);
            gpu_grad_ptr.reset(nullptr);
        }
        device = "cpu";
    }

    void gpu()
    {
        assert(device == "cpu");
        T* p;
        cudaError_t err = cudaMalloc((void**)&p, len * sizeof(T));
        assert(err == cudaSuccess);
        gpu_ptr.reset(p);
        cudaMemcpy(gpu_ptr.get(), cpu_ptr.get(), len * sizeof(T), cudaMemcpyHostToDevice);
        cpu_ptr.reset(nullptr);
        if(requires_grad) 
        {
            T* q;
            cudaError_t err = cudaMalloc((void**)&q, len * sizeof(T));
            assert(err == cudaSuccess);
            gpu_grad_ptr.reset(q);
            cudaMemcpy(gpu_grad_ptr.get(), cpu_grad_ptr.get(), len * sizeof(T), cudaMemcpyHostToDevice);
            cpu_grad_ptr.reset(nullptr);
        }
        device = "gpu";
    }

    void show()
    {
        cout << name << endl;
        printf("device: %s\n", device.c_str());
        printf("shape(");
        for(int i=0;i<num_dim;i++)
        {
            printf("%d", shape[i]);
            if(i!=num_dim-1) printf(", ");
        }
        printf(")\n");

        bool from_gpu = device == "gpu";
        if(from_gpu) cpu();

        cout <<"value:\n";
        int pos[num_dim]{};
        for(int i=0; i<len; i++)
        {
            int idx = pos[0];
            for(int j=1; j<num_dim; j++)
            {
                idx *= shape[j];
                idx += pos[j];
            }
            cout << setw(12) << cpu_ptr[idx] << " ";
            int dim = num_dim - 1;
            pos[dim] ++;
            while(pos[dim] == shape[dim])
            {
                pos[dim] = 0;
                printf("\n");
                dim --;
                pos[dim] ++;
            }
        }
        if(requires_grad)
        {
            cout<<"grad:\n";
            int pos[num_dim]{};
            for(int i=0; i<len; i++)
            {
                int idx = pos[0];
                for(int j=1; j<num_dim; j++)
                {
                    idx *= shape[j];
                    idx += pos[j];
                }
                cout << setw(12) << cpu_grad_ptr[idx] << " ";
                int dim = num_dim - 1;
                pos[dim] ++;
                while(pos[dim] == shape[dim])
                {
                    pos[dim] = 0;
                    printf("\n");
                    dim --;
                    pos[dim] ++;
                }
            }
        }
        if(from_gpu) gpu();
        cout << endl;
    }
};
typedef Tensor_template<float> Tensor;


//relu
__global__ void relu_gpu_forward(float* in, float* out, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

void ReLU_forward(const Tensor& input, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == output.shape.size());
    for(int i=0; i<input.shape.size(); i++)
    {
        assert(input.shape[i] == output.shape[i]);
    }
    relu_gpu_forward<<<CudaGetBlocks(input.len), kCudaThreadsNum>>>(input.gpu_ptr.get(), output.gpu_ptr.get(), input.len);
}

__global__ void relu_gpu_backward(float* in_grad, float* out_value, float* out_grad, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        if(out_value[i] > 0) out_grad[i] += in_grad[i];
    }
}

void ReLU_backward(Tensor& input, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == output.shape.size());
    for(int i=0; i<input.shape.size(); i++)
    {
        assert(input.shape[i] == output.shape[i]);
    }
    relu_gpu_backward<<<CudaGetBlocks(input.len), kCudaThreadsNum>>>(output.gpu_grad_ptr.get(), input.gpu_ptr.get(), input.gpu_grad_ptr.get(), input.len);
}


__global__ void sigmoid_gpu_forward(float* in, float* out, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

void sigmoid_forward(const Tensor& input, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == output.shape.size());
    for(int i=0; i<input.shape.size(); i++)
    {
        assert(input.shape[i] == output.shape[i]);
    }
    sigmoid_gpu_forward<<<CudaGetBlocks(input.len), kCudaThreadsNum>>>(input.gpu_ptr.get(), output.gpu_ptr.get(), input.len);
}

__global__ void sigmoid_gpu_backward(float* in_value, float* in_grad, float* out_grad, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        out_grad[i] += in_grad[i] * in_value[i] * (1 - in_value[i]);
    }
}

void sigmoid_backward(Tensor& input, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == output.shape.size());
    for(int i=0; i<input.shape.size(); i++)
    {
        assert(input.shape[i] == output.shape[i]);
    }
    sigmoid_gpu_backward<<<CudaGetBlocks(input.len), kCudaThreadsNum>>>(output.gpu_ptr.get(), output.gpu_grad_ptr.get(), input.gpu_grad_ptr.get(), input.len);
}

//transpose
//(m, n) -> (n, m)
template<typename T>
__global__ void gpu_transpose(T* x, int s, int t)
{
    int n = s * t;
    CUDA_KERNAL_LOOP(i, n)
    {
        T tmp = x[i];
        __syncthreads();
        int c = i % t;
        int r = i / t;
        x[c*s + r] = tmp;
        __syncthreads();
    }
}

template<typename T>
void transpose(T* x, int m, int n)
{
    gpu_transpose<T><<<CudaGetBlocks(m * n), kCudaThreadsNum>>>(x, m, n);
}

//generate numbers on gpu
template<typename T>
__global__ void same_number_generator(T* numbers, int n, T value)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        numbers[i] = value;
    }
}

template<typename T>
__global__ void uniform_distribution_generator(T *numbers, int n, T lower_bound, T upper_bound, unsigned long long seed)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        curandState state;
        curand_init(seed, i, 0, &state);
        numbers[i] = curand_uniform(&state) * (upper_bound - lower_bound) + lower_bound;
    }
}


//general matrix multiplication
void gemm_gpu(bool transpose_out_put, bool t_a, bool t_b, int m, int n, int k, float alpha, float* A, float* B, float beta, float* C)
{
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_T;
    float* alp = &alpha;
    float* bet = &beta;
    int lda = k, ldb = n, ldc = m;
    if(t_a) 
    {
        transa = CUBLAS_OP_N; lda = m;
    }
    if(t_b) 
    {
        transb = CUBLAS_OP_N; ldb = k;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, transa, transb, m, n, k, alp, A, lda, B, ldb, bet, C, ldc);
    if(transpose_out_put) transpose(C, n, m);
    cublasDestroy(handle);
}


//Fully connected layer
//input(N, C_in) weight(C_in, C_out) bias(1, C_out) output(N, C_out)
const int N = 1000;
Tensor ones_("ones_", {N, 1}, "gpu");

void forward_fc_gpu(float* input, float* output, float* weights, float* bias, int batch_size, int in_features, int out_features)
{
    gemm_gpu(false, 0, 0, batch_size, out_features, in_features, 1, input, weights, 0, output);
    gemm_gpu(true, 0, 0, batch_size, out_features, 1, 1, ones_.gpu_ptr.get(), bias, 1, output);
}

void backward_fc_gpu(float* input, float* output, float* weights, float* bias, int batch_size, int in_features, int out_features, float* grad_output, float* grad_input, float* grad_weights, float* grad_bias)
{
    gemm_gpu(true, 0, 1, batch_size, in_features, out_features, 1, grad_output, weights, 0, grad_input);
    gemm_gpu(true, 1, 0, in_features, out_features, batch_size, 1, input, grad_output, 0, grad_weights);
    gemm_gpu(true, 1, 0, 1, out_features, batch_size, 1, ones_.gpu_ptr.get(), grad_output, 0, grad_bias);
}

void forward_fc(const Tensor& input, const Tensor& weights, const Tensor& bias, Tensor& output)
{
    assert(input.device == "gpu" && weights.device == "gpu" && bias.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == 2 && weights.shape.size() == 2 && bias.shape.size() == 2 && output.shape.size() == 2);
    assert(input.shape[1] == weights.shape[0] && weights.shape[1] == output.shape[1] && input.shape[0] == output.shape[0] && bias.shape[0] == 1 && bias.shape[1] == output.shape[1]);
    same_number_generator<float><<<CudaGetBlocks(ones_.len), kCudaThreadsNum>>>(ones_.gpu_ptr.get(), ones_.len, 1);
    forward_fc_gpu(input.gpu_ptr.get(), output.gpu_ptr.get(), weights.gpu_ptr.get(), bias.gpu_ptr.get(), input.shape[0], input.shape[1], weights.shape[1]);
}

void backward_fc(Tensor& input, Tensor& weights, Tensor& bias, const Tensor& output)
{
    assert(input.device == "gpu" && weights.device == "gpu" && bias.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == 2 && weights.shape.size() == 2 && bias.shape.size() == 2 && output.shape.size() == 2);
    assert(input.shape[1] == weights.shape[0] && weights.shape[1] == output.shape[1] && input.shape[0] == output.shape[0] && bias.shape[0] == 1 && bias.shape[1] == output.shape[1]);
    same_number_generator<float><<<CudaGetBlocks(ones_.len), kCudaThreadsNum>>>(ones_.gpu_ptr.get(), ones_.len, 1);
    backward_fc_gpu(input.gpu_ptr.get(), output.gpu_ptr.get(), weights.gpu_ptr.get(), bias.gpu_ptr.get(), input.shape[0], input.shape[1], output.shape[1], output.gpu_grad_ptr.get(), input.gpu_grad_ptr.get(), weights.gpu_grad_ptr.get(), bias.gpu_grad_ptr.get());
}



//Convolution layer
//input(C_in, H, W)  input_hat(H * W, C_in * K * K)
__global__ void forward_im2col_gpu(float* x, float* x_hat, int H, int W, int C, int K, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        int t = i;
        int kw = t % K;
        t /= K;
        int kh = t % K;
        t /= K;
        int c = t % C;
        t /= C;
        int w = t % W;
        int h = t / W;
        int h1 = h + kh - K/2;
        int w1 = w + kw - K/2;
        if(h1 < 0 || h1 >= H || w1 < 0 || w1 >= W) continue;
        x_hat[i] = x[(c*H + h1)*W + w1];
    }
}
//input(C_in, H, W)  input_hat(H * W, C_in * K * K)
__global__ void backward_col2im_gpu(float* x, float* x_hat_grad ,int H, int W, int C, int K, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        int t = i;
        int kw = t % K;
        t /= K;
        int kh = t % K;
        t /= K;
        int c = t % C;
        t /= C;
        int w = t % W;
        int h = t / W;
        int h1 = h + kh - K/2;
        int w1 = w + kw - K/2;
        if(h1 < 0 || h1 >= H || w1 < 0 || w1 >= W) continue;
        atomicAdd(&x[(c*H + h1)*W + w1], x_hat_grad[i]);
    }
}

//input(C_in, H, W) input_hat(H * W, C_in * K * K)
void forward_im2col(float* x, float* x_hat, int C, int H, int W, int K)
{
    int n = H * W * C * K * K;
    forward_im2col_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(x, x_hat, H, W, C, K, n);
}

void backward_col2im(float* x, float* x_hat, int C, int H, int W, int K)
{
    int n = H * W * C * K * K;
    backward_col2im_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(x, x_hat, H, W, C, K, n);
}


//input(N, C_in, H, W) output(N, C_out, H, W) kernel(C_out, C_in, K, K)
void forward_conv(Tensor& input, Tensor& kernel, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu" && kernel.device == "gpu");
    assert(input.shape.size() == 4 && output.shape.size() == 4 && kernel.shape.size() == 4);
    int N = input.shape[0], C_in = input.shape[1], H = input.shape[2], W = input.shape[3], K = kernel.shape[2], C_out = kernel.shape[0];
    assert(output.shape[0] == N && output.shape[1] == C_out && output.shape[2] == H && output.shape[3] == W && kernel.shape[1] == C_in && kernel.shape[3] == K);
    int n1 = C_in * H * W, n2 = C_out * H * W;
    for(int i=0; i<N; i++)
    {
        Tensor input_hat("input_hat", {H * W, C_in * K * K}, "gpu");
        forward_im2col(input.gpu_ptr.get() + i * n1, input_hat.gpu_ptr.get(), C_in, H, W, K);
        gemm_gpu(true, 0, 1, C_out, H * W, C_in * K * K, 1, kernel.gpu_ptr.get(), input_hat.gpu_ptr.get(), 0, output.gpu_ptr.get() + i * n2);
    }
}
//input(N, C_in, H, W) output(N, C_out, H, W) kernel(C_out, C_in, K, K) input_hat(H * W, C_in * K * K)
void backward_conv(Tensor& input, Tensor& kernel, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu" && kernel.device == "gpu");
    assert(input.shape.size() == 4 && output.shape.size() == 4 && kernel.shape.size() == 4);
    int N = input.shape[0], C_in = input.shape[1], H = input.shape[2], W = input.shape[3], K = kernel.shape[2], C_out = kernel.shape[0];
    assert(output.shape[0] == N && output.shape[1] == C_out && output.shape[2] == H && output.shape[3] == W && kernel.shape[1] == C_in && kernel.shape[3] == K);
    int n1 = C_in * H * W, n2 = C_out * H * W;
    for(int i=0; i<N; i++)
    {
        Tensor input_hat("input_hat", {H * W, C_in * K * K}, "gpu");
        forward_im2col(input.gpu_ptr.get() + i * n1, input_hat.gpu_ptr.get(), C_in, H, W, K);
        //gemm_gpu(true, 0, 0, C_out, C_in * K * K, H * W, 1, output.gpu_grad_ptr.get() + i * n2, input_hat.gpu_ptr.get(), 1, kernel.gpu_grad_ptr.get());
        gemm_gpu(false, 1, 1, C_in * K * K, C_out, H * W, 1, input_hat.gpu_ptr.get(), output.gpu_grad_ptr.get() + i * n2, 1, kernel.gpu_grad_ptr.get());
        gemm_gpu(true, 1, 0, H * W, C_in * K * K, C_out, 1, output.gpu_grad_ptr.get() + i * n2, kernel.gpu_ptr.get(), 1, input_hat.gpu_grad_ptr.get());
        backward_col2im(input.gpu_grad_ptr.get() + i * n1, input_hat.gpu_grad_ptr.get(), C_in, H, W, K);
    }
}



//Pooling Layer
//input(N, C, H, W) output(N, C, H//2, W//2) mask(N, C, H, W)
__global__ void forward_max_pooling_gpu(float* input, float* output, bool* mask, int pool_size, int stride, int N, int C, int H, int W, int h, int w, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        int t = i;
        int w1 = t % w;
        t /= w;
        int h1 = t % h;
        t /= h;
        int c1 = t % C;
        int n1 = t / C;
        float maxn = -1e100;
        int maxh = h1 * stride, maxw = w1 * stride;
        for(int j=0; j<pool_size; j++)
        {
            for(int k=0; k<pool_size; k++)
            {
                int h2 = h1 * stride + j, w2 = w1 * stride + k;
                if(h2 < 0 || h2 >= H || w2 < 0 || w2 >= W) continue;
                float tmp = input[((n1 * C + c1) * H + h2) * W + w2];
                if(tmp > maxn)
                {
                    maxn = tmp;
                    maxh = h2;
                    maxw = w2;
                }
            }
        }
        mask[((n1 * C + c1) * H + maxh) * W + maxw] = 1;
        output[((n1 * C + c1) * h + h1) * w + w1] = maxn;
    }
}

Tensor_template<bool> forward_max_pooling(Tensor& input, Tensor& output, int pool_size, int stride)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == 4 && output.shape.size() == 4);
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    int h = (H + stride - 1) / stride, w = (W + stride - 1) / stride;
    int n = N * C * h * w;
    Tensor_template<bool> mask("mask", {N, C, H, W}, "gpu", 0);
    forward_max_pooling_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input.gpu_ptr.get(), output.gpu_ptr.get(), mask.gpu_ptr.get(), pool_size, stride, N, C, H, W, h, w, n);
    return mask;
}

__global__ void backward_max_pooling_gpu(float* input, float* output, bool* mask, int pool_size, int stride, int N, int C, int H, int W, int h, int w, int n)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        int t = i;
        int w1 = t % w;
        t /= w;
        int h1 = t % h;
        t /= h;
        int c1 = t % C;
        int n1 = t / C;
        for(int j=0; j<pool_size; j++)
        {
            for(int k=0; k<pool_size; k++)
            {
                int pos = ((n1 * C + c1) * H + h1 * stride + j) * W + w1 * stride + k;
                input[pos] = output[i] * mask[pos];
            }
        }        
    }
}

void backward_max_pooling(Tensor& input, Tensor& output, Tensor_template<bool>& mask, int pool_size, int stride)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == 4 && output.shape.size() == 4);
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    int h = (H + stride - 1) / stride, w = (W + stride - 1) / stride;
    int n = N * C * h * w;
    backward_max_pooling_gpu<<<CudaGetBlocks(n), kCudaThreadsNum>>>(input.gpu_grad_ptr.get(), output.gpu_grad_ptr.get(), mask.gpu_ptr.get(), pool_size, stride, N, C, H, W, h, w, n);
}



//Softmax
class SubtractAndExp
{
    public:
        const float a;
        SubtractAndExp(float _a) : a(_a) {}
        __host__ __device__ float operator()(const float& x)
        {
            return expf(x - a);
        }
};

class Division
{
    public:
        const float a;
        Division(float _a) : a(_a) {}
        __host__ __device__ float operator()(const float& x)
        {
            return x / a;
        }
};

//input(N, C) output(N, C)
void softmax(Tensor& input, Tensor& output)
{
    assert(input.device == "gpu" && output.device == "gpu");
    assert(input.shape.size() == 2 && output.shape.size() == 2);
    int N = input.shape[0], C = input.shape[1];
    assert(output.shape[0] == N && output.shape[1] == C);
    thrust::device_vector<float> maxn(N);
    thrust::device_vector<float> sumn(N);
    cudaStream_t* streams = new cudaStream_t[N];
    for(int i=0; i<N; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    for(int i=0; i<N; i++)
    {
        maxn[i] = thrust::reduce(thrust::cuda::par.on(streams[i]), input.gpu_ptr.get() + i * C, input.gpu_ptr.get() + (i+1) * C, -FLT_MAX, thrust::maximum<float>());
    }
    for(int i=0; i<N; i++)
    {
        thrust::transform(thrust::cuda::par.on(streams[i]), input.gpu_ptr.get() + i * C, input.gpu_ptr.get() + (i+1) * C, output.gpu_ptr.get() + i * C, SubtractAndExp(maxn[i]));
    }
    for(int i=0; i<N; i++)
    {
        sumn[i] = thrust::reduce(thrust::cuda::par.on(streams[i]), output.gpu_ptr.get() + i * C, output.gpu_ptr.get() + (i+1) * C, 0.0f, thrust::plus<float>());
    }
    for(int i=0; i<N; i++)
    {
        thrust::transform(thrust::cuda::par.on(streams[i]), output.gpu_ptr.get() + i * C, output.gpu_ptr.get() + (i+1) * C, output.gpu_ptr.get() + i * C, Division(sumn[i]));
    }
    for(int i=0; i<N; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
}



//Cross Entropy Loss
//input(C, ) label(N, ) output(N, )
__global__ void cross_entropy_loss_gpu(float* input, int* label, float* output, int n, int C)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        output[i] = -logf(input[label[i] + i * C]);
    }
}

//input(N, C)  output(N, ) label(N, )
void cross_entropy_loss(Tensor& input, Tensor& output, Tensor_template<int>& label)
{
    assert(input.device == "gpu" && output.device == "gpu" && label.device == "gpu");
    assert(input.shape.size() == 2 && output.shape.size() == 1 && label.shape.size() == 1);
    int N = input.shape[0], C = input.shape[1];
    assert(output.shape[0] == N && label.shape[0] == N);
    cross_entropy_loss_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.gpu_ptr.get(), label.gpu_ptr.get(), output.gpu_ptr.get(), N, C);
}



//backward Cross Entropy Loss with Softmax
//input(N, C) output(N)
__global__ void backward_cross_entropy_loss_with_softmax_gpu(float* input, float* output, float* loss, int* label, int n, int N, int C)
{
    CUDA_KERNAL_LOOP(i, n)
    {
        int c1 = i % C;
        int n1 = i / C;
        input[n1 * C + c1] = loss[n1] * (output[n1 * C + c1] - float(label[n1] == c1));
    }
}

//input(N, C) output(N)
void backward_cross_entropy_loss_with_softmax(Tensor& input, Tensor& output, Tensor& loss, Tensor_template<int>& label)
{
    assert(input.device == "gpu" && output.device == "gpu" && loss.device == "gpu" && label.device == "gpu");
    assert(input.shape.size() == 2 && output.shape.size() == 2 && loss.shape.size() == 1 && label.shape.size() == 1);
    int N = input.shape[0], C = input.shape[1];
    assert(output.shape[0] == N && output.shape[1] == C && loss.shape[0] == N && label.shape[0] == N);
    backward_cross_entropy_loss_with_softmax_gpu<<<CudaGetBlocks(N * C), kCudaThreadsNum>>>(input.gpu_grad_ptr.get(), output.gpu_ptr.get(), loss.gpu_grad_ptr.get(), label.gpu_ptr.get(), N * C, N, C);
}



PYBIND11_MODULE(mytensor, m) {

    //tensor
    py::class_<Tensor>(m,"Tensor")
    .def(py::init<const string &, const vector<int> &, const string &>())
    .def("reshape", &Tensor::reshape)
    .def("shape", &Tensor::size)
    .def("len", &Tensor::length)
    .def("from_np", &Tensor::from_numpy)
    .def("grad_from_np", &Tensor::grad_from_numpy)
    .def("to_np", &Tensor::to_numpy)
    .def("grad_to_np", &Tensor::grad_to_numpy)
    .def("cpu", &Tensor::cpu)
    .def("gpu", &Tensor::gpu)
    .def("show", &Tensor::show);
    
    //tensor int
    py::class_<Tensor_template<int>>(m,"Tensorint")
    .def(py::init<const string &, const vector<int> &, const string &>())
    .def("reshape", &Tensor_template<int>::reshape)
    .def("shape", &Tensor_template<int>::size)
    .def("len", &Tensor_template<int>::length)
    .def("from_np", &Tensor_template<int>::from_numpy)
    .def("grad_from_np", &Tensor_template<int>::grad_from_numpy)
    .def("to_np", &Tensor_template<int>::to_numpy)
    .def("grad_to_np", &Tensor_template<int>::grad_to_numpy)
    .def("cpu", &Tensor_template<int>::cpu)
    .def("gpu", &Tensor_template<int>::gpu)
    .def("show", &Tensor_template<int>::show);

    //tensor bool
    py::class_<Tensor_template<bool>>(m,"Tensorbool")
    .def(py::init<const string &, const vector<int> &, const string &>())
    .def("reshape", &Tensor_template<bool>::reshape)
    .def("shape", &Tensor_template<bool>::size)
    .def("len", &Tensor_template<bool>::length)
    .def("from_np", &Tensor_template<bool>::from_numpy)
    .def("grad_from_np", &Tensor_template<bool>::grad_from_numpy)
    .def("to_np", &Tensor_template<bool>::to_numpy)
    .def("grad_to_np", &Tensor_template<bool>::grad_to_numpy)
    .def("cpu", &Tensor_template<bool>::cpu)
    .def("gpu", &Tensor_template<bool>::gpu)
    .def("show", &Tensor_template<bool>::show);

    //relu
    m.def("relu_forward", &ReLU_forward, "relu forward");
    m.def("relu_backward", &ReLU_backward, "relu backward");

    //sigmoid
    m.def("sigmoid_forward", &sigmoid_forward, "sigmoid forward");
    m.def("sigmoid_backward", &sigmoid_backward, "sigmoid backward");

    //fully connected layer
    m.def("fc_forward", &forward_fc, "fully connected layer forward");
    m.def("fc_backward", &backward_fc, "fully connected layer backward");

    //conv layer
    m.def("conv_forward", &forward_conv, "convolution layer forward");
    m.def("conv_backward", &backward_conv, "convolution layer backward");

    //max pooling
    m.def("max_pooling_forward", &forward_max_pooling, "max pooling layer forward");
    m.def("max_pooling_backward", &backward_max_pooling, "max pooling layer backward");

    //softmax
    m.def("softmax", &softmax, "softmax forward");

    //cross entropy loss
    m.def("cross_entropy_loss", &cross_entropy_loss, "cross entropy loss forward");
    m.def("cross_entropy_loss_with_softmax_backward", &backward_cross_entropy_loss_with_softmax, "cross entropy loss and softmax backward");

}



int main()
{
    same_number_generator<float><<<CudaGetBlocks(ones_.len), kCudaThreadsNum>>>(ones_.gpu_ptr.get(), ones_.len, 1);
}



