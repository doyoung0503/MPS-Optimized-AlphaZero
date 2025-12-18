#pragma once
#include <cstdlib>
#include <cstddef>
#include <memory>
#include <torch/torch.h>

// Custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(void* ptr) const {
        std::free(ptr);
    }
};

// Aligned buffer for Metal Unified Memory optimization
// 16-byte alignment for Metal, 16KB for page alignment
template<typename T, size_t Alignment = 16>
class AlignedBuffer {
public:
    AlignedBuffer() : data_(nullptr), size_(0) {}
    
    explicit AlignedBuffer(size_t count) : size_(count) {
        void* ptr = nullptr;
        size_t bytes = count * sizeof(T);
        
        // Use posix_memalign for aligned allocation
        int result = posix_memalign(&ptr, Alignment, bytes);
        if (result != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        
        data_.reset(static_cast<T*>(ptr));
    }
    
    // Resize buffer
    void resize(size_t count) {
        if (count == size_) return;
        
        void* ptr = nullptr;
        size_t bytes = count * sizeof(T);
        
        int result = posix_memalign(&ptr, Alignment, bytes);
        if (result != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        
        data_.reset(static_cast<T*>(ptr));
        size_ = count;
    }
    
    // Accessors
    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    
    // Create contiguous tensor from this buffer (no copy for MPS)
    torch::Tensor to_tensor(at::IntArrayRef sizes) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        return torch::from_blob(data_.get(), sizes, options);
    }
    
    // Ensure tensor is contiguous and on correct device
    static torch::Tensor prepare_for_gpu(torch::Tensor t, torch::Device device) {
        // Ensure contiguous before GPU transfer
        if (!t.is_contiguous()) {
            t = t.contiguous();
        }
        return t.to(device);
    }
    
private:
    std::unique_ptr<T, AlignedDeleter> data_;
    size_t size_;
};

// Convenience type aliases
using AlignedFloatBuffer = AlignedBuffer<float, 16>;

// Page-aligned buffer for maximum Metal efficiency (16KB alignment)
using PageAlignedFloatBuffer = AlignedBuffer<float, 16384>;
