//
// Created by atharva on 7/11/20.
//

#include "Autoencoder.h"
#include <utility>
#include <cstddef>
#include <iostream>

using namespace std;
VAEOutput VAE::forward(torch::Tensor x) {
    auto encoded = encode(x);
    auto mu = encoded.first;
    auto log_var = encoded.second;
    auto z = reparameterize(mu, log_var);
    auto reconstructed = decode(z);
    return {reconstructed, mu, log_var};
}

VAE::VAE(int64_t image_size, int64_t h_dim, int64_t z_dim):
    fc1(image_size, z_dim),
    fc2(h_dim, z_dim),
    fc3(h_dim, z_dim),
    fc4(z_dim, h_dim),
    fc5(h_dim, h_dim) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
}
std::pair<torch::Tensor, torch::Tensor>VAE::encode(torch::Tensor x) {
    auto h = torch::nn::functional::relu(fc1->forward(x));
    return {fc2->forward(h), fc3->forward(h)};
}

torch::Tensor VAE::reparameterize(torch::Tensor mu, torch::Tensor log_var) {
    if(is_training()){
        auto std = log_var.div(2).exp();
        auto eps = torch::rand_like(std);
        return eps.mul(std).add_(mu);
    }
    else{
        return mu;
    }
}

torch::Tensor VAE::decode(torch::Tensor z) {
    auto h = torch::nn::functional::relu(fc4->forward(z));
    return torch::sigmoid(fc5->forward(h));
}

template<typename dataloader>
void VAE::train_model(int32_t epochs,
        torch::Device device,
        dataloader& Data,
        VAE& model,
        size_t dataset_size,
        std::size_t image_Size) {
    model.train(true);
    model.to(device);
    size_t batch_idx = 0;
    torch::optim::Adam adam(model.parameters(), torch::optim::AdamOptions(0.1));
    for (auto & batch: Data){
        adam.zero_grad();
        auto data = batch.data.to(device).reshape({-1, image_Size});
        auto output = model.forward(data);
        auto reconstruction_loss = torch::nn::functional::binary_cross_entropy(output.reconstruction, data,
                                                                               torch::nn::functional::BinaryCrossEntropyFuncOptions().reduction(torch::kSum));
        auto kl_divergence = -0.5 * torch::sum(1 + output.log_var - output.mu.pow(2) - output.log_var.exp());
        auto loss = reconstruction_loss + kl_divergence;
        AT_ASSERT(!std::nan(loss.template item<float>));
        loss.backward(torch::softmax(loss));
        adam.step();
    }
}
