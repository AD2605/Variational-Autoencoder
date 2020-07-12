#include <torch/torch.h>
#include <utility>

struct VAEOutput {
    torch::Tensor reconstruction;
    torch::Tensor mu;
    torch::Tensor log_var;
};

class VAE : public torch::nn::Module {
public:
    VAE(int64_t image_size, int64_t h_dim, int64_t z_dim);
    torch::Tensor decode(torch::Tensor z);
    VAEOutput forward(torch::Tensor x);

    template<typename dataloader>
    void train_model(int32_t epochs,
            torch::Device device,
            dataloader& data,
            VAE& model,
            size_t dataset_size,
            size_t image_Size);


private:
    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    torch::nn::Linear fc5;
};